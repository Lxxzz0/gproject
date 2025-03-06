task=$1
shots=5
model=huggyllama/llama-7b
model_arch=llama

# Generate the output from LLaMA-7b with H2O
python -u run_helm.py \
  --input_path data/xsum.jsonl \
  --output_path generate_xsum_llama7b_h20.jsonl \
  --model_name ${model} \
  --model_arch ${model_arch} \
  --enable_small_cache \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1

cd helm
TASK=xsum
JSONL=generate_xsum_llama7b.jsonl
OUTPUT=xsum_llama7b_result
ARCH=llama
python scripts/offline_eval/import_results.py together ${JSONL} --cache-dir prod_env/cache
helm-run --conf src/helm/benchmark/presentation/${TASK}/run_specs_${ARCH}.conf --local --max-eval-instances 100 --num-train-trials=1 --suite ${OUTPUT} -n 1
helm-summarize --suite ${OUTPUT} 
# The results are writted into a tex file that can be found in benchmark_output/runs/xsum_llama7b_result/groups/latex/ 