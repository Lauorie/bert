import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from modules.listconranker import ListConRanker

reranker = ListConRanker('/root/app/models/ByteDance-ListConRanker', use_fp16=True, list_transformer_layer=2)

pools = [
    "政治与法律:政策、法律法规、法案等",
    "商业与经济:市场、金融、企业管理、经济分析等",
    "信息技术:软件开发、硬件、网络技术、人工智能等",
    "教育与培训:教学资料、教育研究、课程设计等",
    "健康与医疗:医学研究、健康管理、药物开发等",
    "自然科学:物理、化学、生物、地理等科学研究",
    "文学与艺术:散文、小说、诗歌、戏剧等",
    "社会与文化:社会现象、文化研究、历史等",
    "军事与国防:军事科技、国防政策等",
    "日常生活:家庭、旅游、消费、娱乐等"
]

# [query, passages_1, passage_2, ..., passage_n]
batch = [
    [
        'We propose a Listwise-encoded Contrastive text reRanker (ListConRanker), includes a ListTransformer module for listwise encoding. The ListTransformer can facilitate global contrastive information learning between passage features, including the clustering of similar passages, the clustering between dissimilar passages, and the distinction between similar and dissimilar passages. Besides, we propose ListAttention to help ListTransformer maintain the features of the query while learning global comparative information.', # query
        *pools
    ]
]

# for conventional inference, please manage the batch size by yourself
scores = reranker.compute_score(batch)

# 打印分值最大的pool
max_score_index = scores[0].index(max(scores[0]))
print(max_score_index)
print(f"分值最大的pool: {pools[max_score_index]}")

