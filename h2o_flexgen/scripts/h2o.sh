export HF_HOME="/home/ubuntu/data/gproject/h2o_flexgen/models"

# flexgen
# python flex_opt.py --gpu-batch-size 1 --overlap false --model facebook/opt-6.7b

# h2o
# python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.1 --hh-all --model facebook/opt-6.7b
# python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.2 --hh-all --model facebook/opt-6.7b
# python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.3 --hh-all --model facebook/opt-6.7b

# h2o_compress
# python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.1 --hh-all --model facebook/opt-6.7b --compress-weight
# python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.2 --hh-all --model facebook/opt-6.7b --compress-weight
# python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.3 --hh-all --model facebook/opt-6.7b --compress-weight

# flexgen
python flex_opt.py --gpu-batch-size 4 --overlap false --model facebook/opt-6.7b

# h2o
# python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.1 --hh-all --model facebook/opt-6.7b
# python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.2 --hh-all --model facebook/opt-6.7b
# python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.3 --hh-all --model facebook/opt-6.7b

# h2o_compress
# python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.1 --hh-all --model facebook/opt-6.7b --compress-weight
# python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.2 --hh-all --model facebook/opt-6.7b --compress-weight
# python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.3 --hh-all --model facebook/opt-6.7b --compress-weight
