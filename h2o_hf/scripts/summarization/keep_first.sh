# 跑多轮，看看保留前 i 个 token 的效果
# 新方法，全局优先队列
# local
# # hh_ratio，recent_ratio，window_ratio，token_block_ratio
# bash ./scripts/summarization/eval.sh narrativeqa 0 h2o 5 0 0.3 0.3 0.1 0
# bash ./scripts/summarization/eval.sh qasper 0 h2o 5 0 0.3 0.3 0.1 0
# bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0 0.3 0.3 0.1 0
# bash ./scripts/summarization/eval.sh multifieldqa_en 0 h2o 5 0 0.3 0.3 0.1 0
# bash ./scripts/summarization/eval.sh hotpotqa 0 h2o 5 0 0.3 0.3 0.1 0
# bash ./scripts/summarization/eval.sh 2wikimqa 0 h2o 5 0 0.3 0.3 0.1 0
# bash ./scripts/summarization/eval.sh musique 0 h2o 5 0 0.3 0.3 0.1 0
# bash ./scripts/summarization/eval.sh gov_report 0 h2o 5 0 0.3 0.3 0.1 0
# bash ./scripts/summarization/eval.sh multi_news 0 h2o 5 0 0.3 0.3 0.1 0
# bash ./scripts/summarization/eval.sh triviaqa 0 h2o 5 0 0.3 0.3 0.1 0


# global
# bash ./scripts/summarization/eval.sh narrativeqa 0 h2o 5 0 0.3 1 0.1 0
# bash ./scripts/summarization/eval.sh qasper 0 h2o 5 0 0.3 1 0.1 0
# bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0 0.3 1 0.1 0
# bash ./scripts/summarization/eval.sh multifieldqa_en 0 h2o 5 0 0.3 1 0.1 0
# bash ./scripts/summarization/eval.sh hotpotqa 0 h2o 5 0 0.3 1 0.1 0
# bash ./scripts/summarization/eval.sh 2wikimqa 0 h2o 5 0 0.3 1 0.1 0
# bash ./scripts/summarization/eval.sh musique 0 h2o 5 0 0.3 1 0.1 0
# bash ./scripts/summarization/eval.sh gov_report 0 h2o 5 0 0.3 1 0.1 0
# bash ./scripts/summarization/eval.sh multi_news 0 h2o 5 0 0.3 1 0.1 0
# bash ./scripts/summarization/eval.sh triviaqa 0 h2o 5 0 0.3 1 0.1 0


# h2o
# bash ./scripts/summarization/eval.sh narrativeqa 0 h2o 5 0.1 0.2 0.4 0.1 0
# bash ./scripts/summarization/eval.sh qasper 0 h2o 5 0.1 0.2 0.4 0.1 0
bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0.1 0.2 0.4 0.1 0
bash ./scripts/summarization/eval.sh multifieldqa_en 0 h2o 5 0.1 0.2 0.4 0.1 0
bash ./scripts/summarization/eval.sh hotpotqa 0 h2o 5 0.1 0.2 0.4 0.1 0
bash ./scripts/summarization/eval.sh 2wikimqa 0 h2o 5 0.1 0.2 0.4 0.1 0
bash ./scripts/summarization/eval.sh musique 0 h2o 5 0.1 0.2 0.4 0.1 0
bash ./scripts/summarization/eval.sh gov_report 0 h2o 5 0.1 0.2 0.4 0.1 0
bash ./scripts/summarization/eval.sh multi_news 0 h2o 5 0.1 0.2 0.4 0.1 0
bash ./scripts/summarization/eval.sh triviaqa 0 h2o 5 0.1 0.2 0.4 0.1 0

# 原本的h2o方法
# bash ./scripts/summarization/eval.sh narrativeqa 0 h2o 5 0.1 0.1 0.4 0.1 0
# bash ./scripts/summarization/eval.sh qasper 0 h2o 5 0.1 0.1 0.4 0.1 0
# bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0.1 0.1 0.4 0.1 0
# bash ./scripts/summarization/eval.sh multifieldqa_en 0 h2o 5 0.1 0.1 0.4 0.1 0
# bash ./scripts/summarization/eval.sh hotpotqa 0 h2o 5 0.1 0.1 0.4 0.1 0
# bash ./scripts/summarization/eval.sh 2wikimqa 0 h2o 5 0.1 0.1 0.4 0.1 0
# bash ./scripts/summarization/eval.sh musique 0 h2o 5 0.1 0.1 0.4 0.1 0
# bash ./scripts/summarization/eval.sh gov_report 0 h2o 5 0.1 0.1 0.4 0.1 0
# bash ./scripts/summarization/eval.sh multi_news 0 h2o 5 0.1 0.1 0.4 0.1 0
# bash ./scripts/summarization/eval.sh triviaqa 0 h2o 5 0.1 0.1 0.4 0.1 0

# bash ./scripts/summarization/eval.sh triviaqa 0 lx 5 0.1 0.1 0.4 0.1 0


# narrativeqa
# bash ./scripts/summarization/eval.sh narrativeqa 0 h2o 5 0.1 0.1 1
# bash ./scripts/summarization/eval.sh narrativeqa 0 h2o 5 0.1 0.1 2
# bash ./scripts/summarization/eval.sh narrativeqa 0 h2o 5 0.1 0.1 3
# bash ./scripts/summarization/eval.sh narrativeqa 0 h2o 5 0.1 0.1 4
# bash ./scripts/summarization/eval.sh narrativeqa 0 h2o 5 0.1 0.1 5

# bash ./scripts/summarization/eval.sh narrativeqa 0 h2o 5 0.1 0.1 10
# bash ./scripts/summarization/eval.sh narrativeqa 0 h2o 5 0.1 0.1 20
# bash ./scripts/summarization/eval.sh narrativeqa 0 h2o 5 0.1 0.1 30
# bash ./scripts/summarization/eval.sh narrativeqa 0 h2o 5 0.1 0.1 40
# bash ./scripts/summarization/eval.sh narrativeqa 0 h2o 5 0.1 0.1 50

# qasper
# bash ./scripts/summarization/eval.sh qasper 0 h2o 5 0.1 0.1 1
# bash ./scripts/summarization/eval.sh qasper 0 h2o 5 0.1 0.1 2
# bash ./scripts/summarization/eval.sh qasper 0 h2o 5 0.1 0.1 3
# bash ./scripts/summarization/eval.sh qasper 0 h2o 5 0.1 0.1 4
# bash ./scripts/summarization/eval.sh qasper 0 h2o 5 0.1 0.1 5
# bash ./scripts/summarization/eval.sh qasper 0 h2o 5 0.1 0.1 10
# bash ./scripts/summarization/eval.sh qasper 0 h2o 5 0.1 0.1 20
# bash ./scripts/summarization/eval.sh qasper 0 h2o 5 0.1 0.1 30
# bash ./scripts/summarization/eval.sh qasper 0 h2o 5 0.1 0.1 40
# bash ./scripts/summarization/eval.sh qasper 0 h2o 5 0.1 0.1 50

# bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0.1 0.1 0
# bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0.1 0.1 1
# bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0.1 0.1 2
# bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0.1 0.1 3
# bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0.1 0.1 4
# bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0.1 0.1 5

# bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0.1 0.1 10
# bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0.1 0.1 20
# bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0.1 0.1 30
# bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0.1 0.1 40
# bash ./scripts/summarization/eval.sh xsum 0 h2o 5 0.1 0.1 50

# bash ./scripts/summarization/eval.sh xsum 0 h2o 4 0.1 0.1 200
# bash ./scripts/summarization/eval.sh xsum 0 h2o 4 0.1 0.1 300
# bash ./scripts/summarization/eval.sh xsum 0 h2o 4 0.1 0.1 400
# bash ./scripts/summarization/eval.sh xsum 0 h2o 4 0.1 0.1 500
# bash ./scripts/summarization/eval.sh xsum 0 h2o 4 0.1 0.1 600

# bash ./scripts/summarization/eval.sh xsum 0 h2o 4 0.1 0.1 0
# bash ./scripts/summarization/eval.sh xsum 0 h2o 4 0.1 0.1 1
# bash ./scripts/summarization/eval.sh xsum 0 h2o 4 0.1 0.1 2
# bash ./scripts/summarization/eval.sh xsum 0 h2o 4 0.1 0.1 3
# bash ./scripts/summarization/eval.sh xsum 0 h2o 4 0.1 0.1 4

# bash ./scripts/summarization/eval.sh xsum 3 h2o 4 0.1 0.1 200
# bash ./scripts/summarization/eval.sh xsum 3 h2o 4 0.1 0.1 300
# bash ./scripts/summarization/eval.sh xsum 3 h2o 4 0.1 0.1 400
# bash ./scripts/summarization/eval.sh xsum 3 h2o 4 0.1 0.1 500
# bash ./scripts/summarization/eval.sh xsum 3 h2o 4 0.1 0.1 600

# bash ./scripts/summarization/eval.sh xsum 3 h2o 4 0 0.2 0
# bash ./scripts/summarization/eval.sh xsum 3 h2o 4 0.1 0.1 400
# bash ./scripts/summarization/eval.sh xsum 3 h2o 4 0.1 0.1 500
# bash ./scripts/summarization/eval.sh xsum 3 h2o 4 0.1 0.1 600
