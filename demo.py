import io
import json
import re
import sys
import time
import zipfile
import traceback
from pathlib import Path

import gradio as gr
import torch
import torch_npu
import transformers

import modules.chat as chat
import modules.extensions as extensions_module
import modules.shared as shared
import modules.ui as ui
from modules.html_generator import generate_chat_html
from modules.LoRA import add_lora_to_model
from modules.models import load_model, load_soft_prompt
from modules.text_generation import clear_torch_cache, generate_reply, formatted_outputs
from modules.extensions import apply_extensions
from modules.callbacks import (Iteratorize, Stream,
                               _SentinelTokenStoppingCriteria)

# Loading custom settings
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)
torch.npu.set_device(7)
settings_file = None
if shared.args.settings is not None and Path(shared.args.settings).exists():
    settings_file = Path(shared.args.settings)
elif Path('settings.json').exists():
    settings_file = Path('settings.json')
if settings_file is not None:
    print(f"Loading settings from {settings_file}...")
    new_settings = json.loads(open(settings_file, 'r').read())
    for item in new_settings:
        shared.settings[item] = new_settings[item]

def get_available_models():
    if shared.args.flexgen:
        return sorted([re.sub('-np$', '', item.name) for item in list(Path('models/').glob('*')) if item.name.endswith('-np')], key=str.lower)
    else:
        return sorted([re.sub('.pth$', '', item.name) for item in list(Path('models/').glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=str.lower)

def get_available_presets():
    return sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('presets').glob('*.txt'))), key=str.lower)

def get_available_characters():
    return ['None'] + sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('characters').glob('*.json'))), key=str.lower)

def get_available_extensions():
    return sorted(set(map(lambda x : x.parts[1], Path('extensions').glob('*/script.py'))), key=str.lower)

def get_available_softprompts():
    return ['None'] + sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('softprompts').glob('*.zip'))), key=str.lower)

def get_available_loras():
    return ['None'] + sorted([item.name for item in list(Path('loras/').glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=str.lower)

def load_model_wrapper(selected_model):
    if selected_model != shared.model_name:
        shared.model_name = selected_model
        shared.model = shared.tokenizer = None
        clear_torch_cache()
        shared.model, shared.tokenizer = load_model(shared.model_name)

    return selected_model

def load_lora_wrapper(selected_lora):
    add_lora_to_model(selected_lora)
    default_text = shared.settings['lora_prompts'][next((k for k in shared.settings['lora_prompts'] if re.match(k.lower(), shared.lora_name.lower())), 'default')]

    return selected_lora, default_text

def load_preset_values(preset_menu, return_dict=False):
    generate_params = {
        'do_sample': True,
        'temperature': 1,
        'top_p': 1,
        'typical_p': 1,
        'repetition_penalty': 1,
        'encoder_repetition_penalty': 1,
        'top_k': 50,
        'num_beams': 1,
        'penalty_alpha': 0,
        'min_length': 0,
        'length_penalty': 1,
        'no_repeat_ngram_size': 0,
        'early_stopping': False,
    }
    with open(Path(f'presets/{preset_menu}.txt'), 'r') as infile:
        preset = infile.read()
    for i in preset.splitlines():
        i = i.rstrip(',').strip().split('=')
        if len(i) == 2 and i[0].strip() != 'tokens':
            generate_params[i[0].strip()] = eval(i[1].strip())

    generate_params['temperature'] = min(1.99, generate_params['temperature'])

    if return_dict:
        return generate_params
    else:
        return preset_menu, generate_params['do_sample'], generate_params['temperature'], generate_params['top_p'], generate_params['typical_p'], generate_params['repetition_penalty'], generate_params['encoder_repetition_penalty'], generate_params['top_k'], generate_params['min_length'], generate_params['no_repeat_ngram_size'], generate_params['num_beams'], generate_params['penalty_alpha'], generate_params['length_penalty'], generate_params['early_stopping']

def upload_soft_prompt(file):
    with zipfile.ZipFile(io.BytesIO(file)) as zf:
        zf.extract('meta.json')
        j = json.loads(open('meta.json', 'r').read())
        name = j['name']
        Path('meta.json').unlink()

    with open(Path(f'softprompts/{name}.zip'), 'wb') as f:
        f.write(file)

    return name

def create_model_and_preset_menus():
    with gr.Row():
        with gr.Column():
            with gr.Row():
                shared.gradio['model_menu'] = gr.Dropdown(choices=available_models, value=shared.model_name, label='Model')
                ui.create_refresh_button(shared.gradio['model_menu'], lambda : None, lambda : {'choices': get_available_models()}, 'refresh-button')
        with gr.Column():
            with gr.Row():
                shared.gradio['preset_menu'] = gr.Dropdown(choices=available_presets, value=default_preset if not shared.args.flexgen else 'Naive', label='Generation parameters preset')
                ui.create_refresh_button(shared.gradio['preset_menu'], lambda : None, lambda : {'choices': get_available_presets()}, 'refresh-button')

def create_settings_menus(default_preset):
    generate_params = load_preset_values(default_preset if not shared.args.flexgen else 'Naive', return_dict=True)

    with gr.Row():
        with gr.Column():
            with gr.Box():
                gr.Markdown('Custom generation parameters ([reference](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig))')
                with gr.Row():
                    with gr.Column():
                        shared.gradio['temperature'] = gr.Slider(0.01, 1.99, value=generate_params['temperature'], step=0.01, label='temperature')
                        shared.gradio['top_p'] = gr.Slider(0.0,1.0,value=generate_params['top_p'],step=0.01,label='top_p')
                        shared.gradio['top_k'] = gr.Slider(0,200,value=generate_params['top_k'],step=1,label='top_k')
                        shared.gradio['typical_p'] = gr.Slider(0.0,1.0,value=generate_params['typical_p'],step=0.01,label='typical_p')
                    with gr.Column():
                        shared.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=generate_params['repetition_penalty'],step=0.01,label='repetition_penalty')
                        shared.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=generate_params['encoder_repetition_penalty'],step=0.01,label='encoder_repetition_penalty')
                        shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='no_repeat_ngram_size')
                        shared.gradio['min_length'] = gr.Slider(0, 2000, step=1, value=generate_params['min_length'] if shared.args.no_stream else 0, label='min_length', interactive=shared.args.no_stream)
                shared.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='do_sample')
        with gr.Column():
            with gr.Box():
                gr.Markdown('Contrastive search')
                shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='penalty_alpha')

            with gr.Box():
                gr.Markdown('Beam search (uses a lot of VRAM)')
                with gr.Row():
                    with gr.Column():
                        shared.gradio['num_beams'] = gr.Slider(1, 20, step=1, value=generate_params['num_beams'], label='num_beams')
                    with gr.Column():
                        shared.gradio['length_penalty'] = gr.Slider(-5, 5, value=generate_params['length_penalty'], label='length_penalty')
                shared.gradio['early_stopping'] = gr.Checkbox(value=generate_params['early_stopping'], label='early_stopping')

            shared.gradio['seed'] = gr.Number(value=-1, label='Seed (-1 for random)')

    with gr.Row():
        shared.gradio['preset_menu_mirror'] = gr.Dropdown(choices=available_presets, value=default_preset if not shared.args.flexgen else 'Naive', label='Generation parameters preset')
        ui.create_refresh_button(shared.gradio['preset_menu_mirror'], lambda : None, lambda : {'choices': get_available_presets()}, 'refresh-button')

    with gr.Row():
        shared.gradio['lora_menu'] = gr.Dropdown(choices=available_loras, value=shared.lora_name, label='LoRA')
        ui.create_refresh_button(shared.gradio['lora_menu'], lambda : None, lambda : {'choices': get_available_loras()}, 'refresh-button')

    with gr.Accordion('Soft prompt', open=False):
        with gr.Row():
            shared.gradio['softprompts_menu'] = gr.Dropdown(choices=available_softprompts, value='None', label='Soft prompt')
            ui.create_refresh_button(shared.gradio['softprompts_menu'], lambda : None, lambda : {'choices': get_available_softprompts()}, 'refresh-button')

        gr.Markdown('Upload a soft prompt (.zip format):')
        with gr.Row():
            shared.gradio['upload_softprompt'] = gr.File(type='binary', file_types=['.zip'])

    shared.gradio['model_menu'].change(load_model_wrapper, [shared.gradio['model_menu']], [shared.gradio['model_menu']], show_progress=True)
    shared.gradio['preset_menu'].change(load_preset_values, [shared.gradio['preset_menu']], [shared.gradio[k] for k in ['preset_menu_mirror', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']])
    shared.gradio['preset_menu_mirror'].change(load_preset_values, [shared.gradio['preset_menu_mirror']], [shared.gradio[k] for k in ['preset_menu', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']])
    shared.gradio['lora_menu'].change(load_lora_wrapper, [shared.gradio['lora_menu']], [shared.gradio['lora_menu'], shared.gradio['textbox']], show_progress=True)
    shared.gradio['softprompts_menu'].change(load_soft_prompt, [shared.gradio['softprompts_menu']], [shared.gradio['softprompts_menu']], show_progress=True)
    shared.gradio['upload_softprompt'].upload(upload_soft_prompt, [shared.gradio['upload_softprompt']], [shared.gradio['softprompts_menu']])

def set_interface_arguments(interface_mode, extensions, cmd_active):
    modes = ["default", "notebook", "chat", "cai_chat"]
    cmd_list = vars(shared.args)
    cmd_list = [k for k in cmd_list if type(cmd_list[k]) is bool and k not in modes]

    shared.args.extensions = extensions
    for k in modes[1:]:
        exec(f"shared.args.{k} = False")
    if interface_mode != "default":
        exec(f"shared.args.{interface_mode} = True")

    for k in cmd_list:
        exec(f"shared.args.{k} = False")
    for k in cmd_active:
        exec(f"shared.args.{k} = True")

    shared.need_restart = True

available_models = get_available_models()
available_presets = get_available_presets()
available_characters = get_available_characters()
available_softprompts = get_available_softprompts()
available_loras = get_available_loras()

# Default extensions
extensions_module.available_extensions = get_available_extensions()
if shared.args.chat or shared.args.cai_chat:
    for extension in shared.settings['chat_default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)
else:
    for extension in shared.settings['default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)

# Default model
if shared.args.model is not None:
    shared.model_name = shared.args.model
else:
    if len(available_models) == 0:
        print('No models are available! Please download at least one.')
        sys.exit(0)
    elif len(available_models) == 1:
        i = 0
    else:
        print('The following models are available:\n')
        for i, model in enumerate(available_models):
            print(f'{i+1}. {model}')
        print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
        i = int(input())-1
        print()
    shared.model_name = available_models[i]
shared.model, shared.tokenizer = load_model(shared.model_name)
if shared.args.lora:
    print(shared.args.lora)
    shared.lora_name = shared.args.lora
    add_lora_to_model(shared.lora_name)

# Default UI settings
default_preset = shared.settings['presets'][next((k for k in shared.settings['presets'] if re.match(k.lower(), shared.model_name.lower())), 'default')]
default_text = shared.settings['lora_prompts'][next((k for k in shared.settings['lora_prompts'] if re.match(k.lower(), shared.lora_name.lower())), 'default')]
if default_text == '':
    default_text = shared.settings['prompts'][next((k for k in shared.settings['prompts'] if re.match(k.lower(), shared.model_name.lower())), 'default')]
title ='Text generation web UI'
description = '\n\n# Text generation lab\nGenerate text using Large Language Models.\n'
suffix = '_pygmalion' if 'pygmalion' in shared.model_name.lower() else ''
def set_manual_seed(seed):
    if seed != -1:
        torch.manual_seed(seed)
        if torch.npu.is_available():
            # torch.npu.set_compile_mode(jit_compile=False)
            torch.npu.manual_seed_all(seed)

def encode(prompt, tokens_to_generate=0, add_special_tokens=True):
    if shared.is_RWKV:
        input_ids = shared.tokenizer.encode(str(prompt))
        input_ids = np.array(input_ids).reshape(1, len(input_ids))
        return input_ids
    else:
        input_ids = shared.tokenizer.encode(str(prompt), return_tensors='pt', truncation=True, max_length=get_max_prompt_length(tokens_to_generate), add_special_tokens=add_special_tokens)
        if shared.args.cpu:
            return input_ids
        elif shared.args.flexgen:
            return input_ids.numpy()
        elif shared.args.deepspeed:
            return input_ids.to(device=local_rank)
        elif 0: #torch.has_mps:
            device = torch.device('mps')
            return input_ids.to(device)
        else:
            return input_ids.npu()
        
def decode(output_ids):
    # Open Assistant relies on special tokens like <|endoftext|>
    if re.match('(oasst|galactica)-*', shared.model_name.lower()):
        return shared.tokenizer.decode(output_ids, skip_special_tokens=False)
    else:
        reply = shared.tokenizer.decode(output_ids, skip_special_tokens=True)
        reply = reply.replace(r'<|endoftext|>', '')
        return reply
        
def get_max_prompt_length(tokens):
    max_length = 2048-tokens
    if shared.soft_prompt:
        max_length -= shared.soft_prompt_tensor.shape[1]
    return max_length

def clear_torch_cache():
    if not shared.args.cpu:
        torch_npu.npu.empty_cache()

def data_to_device(samples):
    """Return samples from host to device."""
    if not torch.npu.is_available():
        return samples
    if isinstance(samples, list) or isinstance(samples, tuple):
        for i in range(len(samples)):
            samples[i] = data_to_device(samples[i])
    elif isinstance(samples, dict):
        for k, v in samples.items():
            samples[k] = data_to_device(v)
    elif torch.is_tensor(samples):
        samples = samples.npu()
    
    else:
        samples = samples
    return samples

def reply(question, max_new_tokens, do_sample, temperature, top_p, typical_p, repetition_penalty, encoder_repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, seed, eos_token=None, stopping_strings=[]):
    clear_torch_cache()
    set_manual_seed(seed)
    t0 = time.time()
    
    original_question = question
    print(f"\n\n{question}\n--------------------\n")
    input_ids = shared.tokenizer.encode(str(question), 
                                        return_tensors='pt', truncation=True, max_length=get_max_prompt_length(max_new_tokens), add_special_tokens=True)
    if shared.args.cpu:
        input_ids
    elif 0: #torch.has_mps:
        device = torch.device('mps')
        input_ids.to(device)
    else:
        input_ids.npu()
        
    original_input_ids = input_ids
    output = input_ids[0]

    cuda = not any((shared.args.cpu, shared.args.deepspeed, shared.args.flexgen))
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    if eos_token is not None:
        eos_token_ids.append(int(encode(eos_token)[0][-1]))
    stopping_criteria_list = transformers.StoppingCriteriaList()
    generate_params = {}
    generate_params.update({
            "max_new_tokens": max_new_tokens,
            "eos_token_id": eos_token_ids,
            "stopping_criteria": stopping_criteria_list,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "typical_p": typical_p,
            "repetition_penalty": repetition_penalty,
            "encoder_repetition_penalty": encoder_repetition_penalty,
            "top_k": top_k,
            "min_length": min_length if shared.args.no_stream else 0,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "num_beams": num_beams,
            "penalty_alpha": penalty_alpha,
            "length_penalty": length_penalty,
            "early_stopping": early_stopping,
        })
    generate_params.update({"inputs": input_ids})
    generate_params = data_to_device(generate_params)
    try:
        # Generate the entire reply at once.
        if shared.args.no_stream:
            with torch.no_grad():
                #with torch.npu.profile("llama_sta"):
                output = shared.model.generate(**generate_params)[0]
                if cuda:
                    output = output.npu()

            new_tokens = len(output) - len(input_ids[0])
            reply = decode(output[-new_tokens:])
            if not (shared.args.chat or shared.args.cai_chat):
                reply = original_question + apply_extensions(reply, "output")

            # yield formatted_outputs(reply, shared.model_name)
            print(reply)
            out = formatted_outputs(reply, shared.model_name)
            return out
        elif not shared.args.flexgen:

            def generate_with_callback(callback=None, **kwargs):
                kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                clear_torch_cache()
                with torch.no_grad():
                    shared.model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, kwargs, callback=None)

            if not (shared.args.chat or shared.args.cai_chat):
                print(original_question)
            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    new_tokens = len(output) - len(input_ids[0])
                    reply = decode(output[-new_tokens:])
                    if not (shared.args.chat or shared.args.cai_chat):
                        reply = original_question + apply_extensions(reply, "output")

                    if output[-1] in eos_token_ids:
                        break
                    print(reply)

                print(reply)

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        print(f"Output generated in {(t1-t0):.2f} seconds ({(len(output)-len(original_input_ids[0]))/(t1-t0):.2f} tokens/s, {len(output)-len(original_input_ids[0])} tokens)")
        return

reply(question='Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:', 
                max_new_tokens=200, 
                do_sample=True, 
                temperature=1.99, 
                top_p=0.18, 
                typical_p=1, 
                repetition_penalty=1.15, 
                encoder_repetition_penalty=1, 
                top_k=30, 
                min_length=0, 
                no_repeat_ngram_size=0, 
                num_beams=1, 
                penalty_alpha=0, 
                length_penalty=1, 
                early_stopping=False, 
                seed=-1, 
                eos_token=None, 
                stopping_strings=[])


