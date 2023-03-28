source /usr/local/Ascend/ascend-toolkit/set_env.sh
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
#设置默认日志级别,0-debug/1-info/2-warning/3-error
# export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_DEVICE_ID=7
export DEVICE_ID=7
python3 demo.py --model llama-7b --no-stream
# python3 server.py --model llama-7b #--no-stream
