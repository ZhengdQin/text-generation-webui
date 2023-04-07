# #change the second input of max op to fp16 type
# import torch
# import torch_npu
# from auto_optimizer import OnnxGraph
# import onnx
# import os
# onnx_graph = OnnxGraph.parse("./model.onnx")
# for max_node in onnx_graph.get_nodes(op_type="Max"):
#     max_init = onnx_graph[max_node.inputs[1]]
#     max_init.value = max_init.value.astype("float16")
# path = "./model_trans.onnx"
# breakpoint()
# onnx.save(onnx_graph.model(),
#             path, 
#             save_as_external_data=True,
#             all_tensors_to_one_file=True,
#             location=os.path.basename(path) + '.data')
# onnx_graph.save("./model_trans.onnx")
# #==========================================================

from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForCausalLM
from pathlib import Path
import torch
import torch_npu
import onnxruntime
import onnx

torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)
torch.npu.set_device(7)
tokenizer = AutoTokenizer.from_pretrained("./models/llama-7b")

# #======================onnx runtine===================
# inputs = tokenizer("How to learn a language", return_tensors="pt")
# # sess = onnxruntime.InferenceSession("./models/llama-7b/model_trans.onnx")

# # print(f"input length: {le
# # utputs()[0].name}, shape: {sess.get_outputs()[0].shape}")
# sess_input = {'input_ids':inputs['input_ids'].numpy(), 'attention_mask':inputs['input_ids'].numpy()}
# output = sess.run(['logits'],sess_input)
# #======================================================

#======================hugging face====================
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

model = ORTModelForCausalLM.from_pretrained("./models/llama-7b", file_name="model.onnx",use_cache=False)
inputs = tokenizer(['Common sense questions and answers\n\nQuestion: How to learn a new language?\nFactual answer:'], return_tensors="pt")
data_to_device(inputs)
gen_tokens = model.generate(**inputs, do_sample=True,temperature=1.9, min_length=40, max_length=40)
# gen_tokens = model.generate(**inputs, do_sample=True,temperature=1.99, top_p=0.18, typical_p=1, repetition_penalty=1.15, encoder_repetition_penalty=1, top_k=30, min_length=0, no_repeat_ngram_size=0, num_beams=1, penalty_alpha=0, length_penalty=1, early_stopping=False, max_length=40)
print(tokenizer.batch_decode(gen_tokens))