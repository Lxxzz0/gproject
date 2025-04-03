# 倒数第三个参数是GPU的编号
# 跑 h2o
bash ./scripts/summarization/eval.sh xsum 0 h2o 3 1024 1024

bash ./scripts/summarization/eval.sh xsum 0 h2o 3 4 512

bash ./scripts/summarization/eval.sh xsum 3 h2o 3 4 512

bash ./scripts/summarization/eval.sh xsum 3 h2o 3 8 512

bash ./scripts/summarization/eval.sh xsum 3 h2o 0 1024 1024

# h2o
bash ./scripts/summarization/eval.sh xsum 3 h2o 0 0.1 0.1

# full
bash ./scripts/summarization/eval.sh xsum 3 full 3 0 1

# local and random，因为改动都是在 kv cache 剪枝的模块上做的，暂时没改其他参数
bash ./scripts/summarization/eval.sh xsum 3 h2o 3 0 0.2
