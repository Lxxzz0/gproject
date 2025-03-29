## LLaMA-7B
bash scripts/lm_eval/full_cache.sh openbookqa huggyllama/llama-7b llama choice
bash scripts/lm_eval/h2o.sh openbookqa huggyllama/llama-7b llama choice
bash scripts/lm_eval/local.sh openbookqa huggyllama/llama-7b llama choice

bash scripts/lm_eval/full_cache.sh copa huggyllama/llama-7b llama
bash scripts/lm_eval/h2o.sh copa huggyllama/llama-7b llama
bash scripts/lm_eval/local.sh copa huggyllama/llama-7b llama

bash scripts/lm_eval/full_cache.sh piqa huggyllama/llama-7b llama
bash scripts/lm_eval/h2o.sh piqa huggyllama/llama-7b llama
bash scripts/lm_eval/local.sh piqa huggyllama/llama-7b llama

bash scripts/lm_eval/full_cache.sh winogrande huggyllama/llama-7b llama
bash scripts/lm_eval/h2o.sh winogrande huggyllama/llama-7b llama
bash scripts/lm_eval/local.sh winogrande huggyllama/llama-7b llama

bash scripts/lm_eval/full_cache.sh mathqa huggyllama/llama-7b llama
bash scripts/lm_eval/h2o.sh mathqa huggyllama/llama-7b llama
bash scripts/lm_eval/local.sh mathqa huggyllama/llama-7b llama

bash scripts/lm_eval/full_cache.sh rte huggyllama/llama-7b llama
bash scripts/lm_eval/h2o.sh rte huggyllama/llama-7b llama
bash scripts/lm_eval/local.sh rte huggyllama/llama-7b llama

# test
bash scripts/lm_eval/test.sh openbookqa huggyllama/llama-7b llama

# generate
bash scripts/lm_eval/full_cache.sh openbookqa huggyllama/llama-7b llama generate
bash scripts/lm_eval/h2o.sh openbookqa huggyllama/llama-7b llama generate
bash scripts/lm_eval/local.sh openbookqa huggyllama/llama-7b llama generate
