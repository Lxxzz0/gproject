# 定义数据盘根目录
DATA_ROOT="/home/ubuntu/data"

# 定义子目录
HF_CACHE="${DATA_ROOT}/hf_cache"      # Hugging Face 缓存
TASK_DATA="${DATA_ROOT}/task_data"    # 任务数据
TASK_RESULTS="${DATA_ROOT}/results"   # 任务输出结果

# 创建目录（如果不存在）
mkdir -p ${HF_CACHE} ${TASK_DATA} ${TASK_RESULTS}

# 设置 Hugging Face 缓存路径
export HF_HOME=${HF_CACHE}

# ## Obtain inference data
task=$1
shots=5
python -u generate_task_data.py --output-file ${TASK_DATA}/${task}-${shots}.jsonl --task-name ${task} --num-fewshot ${shots}

## Inference, and generate output json file
model=$2
model_arch=$3
python -u run_lm_eval_harness.py --input-path ${TASK_DATA}/${task}-${shots}.jsonl --output-path ${TASK_RESULTS}/${task}-${shots}-${model_arch}-local.jsonl --model-name ${model} --model-type ${model_arch} --heavy_ratio 0 --recent_ratio 0.2 --enable_small_cache

## Evaluate results
python -u evaluate_task_result.py --result-file ${TASK_RESULTS}/${task}-${shots}-${model_arch}-local.jsonl --task-name ${task} --num-fewshot ${shots} --model-type ${model_arch}




