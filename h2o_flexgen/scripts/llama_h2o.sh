# export HF_HOME="/home/ubuntu/data/hf_cache"
# 设置 PYTHONPATH，确保 Python 能找到 flexgen_llama 模块
export PYTHONPATH=$PYTHONPATH:~/data/H2O-main/h2o_flexgen


# flexgen
# python flex_llama.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b

# h2o
# python flex_llama.py --gpu-batch-size 1 --overlap false --hh-ratio 0.1 --hh-all --model facebook/opt-6.7b
# python flex_llama.py --gpu-batch-size 1 --overlap false --hh-ratio 0.2 --hh-all --model facebook/opt-6.7b
# python flex_llama.py --gpu-batch-size 1 --overlap false --hh-ratio 0.3 --hh-all --model facebook/opt-6.7b

# h2o_compress
# python flex_llama.py --gpu-batch-size 1 --overlap false --hh-ratio 0.1 --hh-all --model facebook/opt-6.7b --compress-weight
# python flex_llama.py --gpu-batch-size 1 --overlap false --hh-ratio 0.2 --hh-all --model facebook/opt-6.7b --compress-weight
# python flex_llama.py --gpu-batch-size 1 --overlap false --hh-ratio 0.3 --hh-all --model facebook/opt-6.7b --compress-weight

# flexgen
python flex_llama.py --gpu-batch-size 64 --overlap false --model huggyllama/llama-7b

# h2o
# python flex_llama.py --gpu-batch-size 1 --overlap false --hh-ratio 0.1 --hh-all --model facebook/opt-6.7b
# python flex_llama.py --gpu-batch-size 1 --overlap false --hh-ratio 0.2 --hh-all --model facebook/opt-6.7b
# python flex_llama.py --gpu-batch-size 1 --overlap false --hh-ratio 0.3 --hh-all --model facebook/opt-6.7b

# h2o_compress
# python flex_llama.py --gpu-batch-size 1 --overlap false --hh-ratio 0.1 --hh-all --model facebook/opt-6.7b --compress-weight
# python flex_llama.py --gpu-batch-size 1 --overlap false --hh-ratio 0.2 --hh-all --model facebook/opt-6.7b --compress-weight
# python flex_llama.py --gpu-batch-size 1 --overlap false --hh-ratio 0.3 --hh-all --model facebook/opt-6.7b --compress-weight
