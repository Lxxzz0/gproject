task=$1
model=$2
model_arch=$3

if [[ ${model_arch} == 'opt' ]]; then
    input_data=data/${task}_opt.jsonl
else
    input_data=data/${task}.jsonl
fi

# python -u run_helm.py \
#     --input_path ${input_data} \
#     --output_path ${task}-${model_arch}-full.jsonl \
#     --model_name ${model} \
#     --model_arch ${model_arch}

## Evaluate results
python -u evaluate_task_result.py --result-file ${task}-${shots}-${model_arch}-full.jsonl --task-name ${task} --num-fewshot ${shots} --model-type ${model_arch}
