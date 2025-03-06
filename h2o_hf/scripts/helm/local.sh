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
#     --output_path ${task}-${model_arch}-local.jsonl \
#     --model_name ${model} \
#     --model_arch ${model_arch} \
#     --enable_small_cache \
#     --heavy_ratio 0 \
#     --recent_ratio 0.2 

cd helm
TASK=xsum
JSONL=generate_xsum_llama7b.jsonl
OUTPUT=xsum_llama7b_result
ARCH=llama
python scripts/offline_eval/import_results.py together ${JSONL} --cache-dir prod_env/cache
helm-run --conf src/helm/benchmark/presentation/${TASK}/run_specs_${ARCH}.conf --local --max-eval-instances 100 --num-train-trials=1 --suite ${OUTPUT} -n 1
helm-summarize --suite ${OUTPUT} 
# The results are writted into a tex file that can be found in benchmark_output/runs/xsum_llama7b_result/groups/latex/ 