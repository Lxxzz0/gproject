# 生成提示信息
bash scripts/generation/llama_local.sh

# 生成回复

# llama-7b openbookqa
bash scripts/lm_eval/local.sh openbookqa huggyllama/llama-7b llama
bash scripts/lm_eval/full_cache.sh openbookqa huggyllama/llama-7b llama
bash scripts/lm_eval/h2o.sh openbookqa huggyllama/llama-7b llama
# llama-7b copa
bash scripts/lm_eval/full_cache.sh copa huggyllama/llama-7b llama
bash scripts/lm_eval/h2o.sh copa huggyllama/llama-7b llama
bash scripts/lm_eval/local.sh copa huggyllama/llama-7b llama


