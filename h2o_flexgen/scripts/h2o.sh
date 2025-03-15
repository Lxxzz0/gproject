export HF_HOME="/home/ubuntu/data/h2o_flexgen/models"


# python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.1 --hh-all --model facebook/opt-6.7b
python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.2 --hh-all --model facebook/opt-6.7b
# python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.3 --hh-all --model facebook/opt-6.7b