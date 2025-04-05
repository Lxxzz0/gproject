# run : bash scripts/summarization/eval.sh xsum 0 h2o 5 0.1 0.1 0

# 定义数据盘根目录
DATA_ROOT="/home/ubuntu/data/tmp/gproject/h2o_hf"

# 定义子目录
HF_CACHE="${DATA_ROOT}/hf_cache"      # Hugging Face 缓存
TASK_DATA="${DATA_ROOT}/task_data"    # 任务数据
TASK_RESULTS="${DATA_ROOT}/results"   # 任务输出结果

# 创建目录（如果不存在）
mkdir -p ${HF_CACHE} ${TASK_DATA} ${TASK_RESULTS}

# 设置 Hugging Face 缓存路径
export HF_HOME=${HF_CACHE}

task=$1
shots=$2
method=$3
GPU=$4
# HH_SIZE=$5
# RECENT_SIZE=$6
hh_ratio=$5
recent_ratio=$6
keep_first=$7

if [[ ${method} == 'h2o' ]]; then
    CUDA_VISIBLE_DEVICES=${GPU} python -u run_summarization.py \
        --dataset ${task} \
        --input_path data/summarization_data/${task}_${shots}shot.jsonl \
        --output_path summary_results/${task}_${shots}shot_h2o_hh${1}_${2}.jsonl \
        --model_name /home/ubuntu/data/models/huggyllama-llama-7b \
        --hh_ratio ${hh_ratio} \
        --recent_ratio ${recent_ratio} \
        --cache_dir ${HF_CACHE}  \
        --keep_first ${keep_first} \
        --enable_h2o_cache
elif [[ ${method} == 'full' ]]; then
    CUDA_VISIBLE_DEVICES=${GPU} python -u run_summarization.py \
        --input_path data/summarization_data/${task}_${shots}shot.jsonl \
        --output_path summary_results/${task}_${shots}shot_full.jsonl \
        --model_name /home/ubuntu/data/models/huggyllama-llama-7b \
        --hh_ratio ${hh_ratio} \
        --recent_ratio ${recent_ratio} \
        --cache_dir ${HF_CACHE} 
else    # local
    CUDA_VISIBLE_DEVICES=${GPU} python -u run_summarization.py \
        --input_path data/summarization_data/${task}_${shots}shot.jsonl \
        --output_path summary_results/${task}_${shots}shot_local.jsonl \
        --model_name /home/ubuntu/data/models/huggyllama-llama-7b \
        --cache_dir ${HF_CACHE} \
        --recent_ratio ${recent_ratio} \
        --enable_h2o_cache
fi
