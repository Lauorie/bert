import os
import re
import json
from tqdm import tqdm
from loguru import logger
from langdetect import detect, DetectorFactory

from modules.listconranker import ListConRanker



# 设置 CUDA_VISIBLE_DEVICES 环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 初始化 reranker 对象
reranker = ListConRanker('/root/app/models/ByteDance-ListConRanker', use_fp16=True, list_transformer_layer=2)

# 定义领域池
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

# 编译正则表达式，用于清理文本（移除数字和非单词字符）
pattern = re.compile(r'[\d\W]')

# 定义语言映射表
lang_map = {
    'en': 'English',
    'zh-cn': 'Chinese',
    'zh-tw': 'Chinese', 
    'ja': 'Japanese',
    'ko': 'Korean',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish'
}

# 全局设置seed，确保结果一致
DetectorFactory.seed = 0

def detect_language(text):
    """
    检测文本语言，若文本为空则默认返回 "English"
    """
    cleaned_text = pattern.sub('', text)
    if not cleaned_text:
        return "English"
    
    try:
        lang = detect(cleaned_text)
        return lang_map.get(lang, "English")
    except Exception as e:
        logger.error(f"语言检测错误: {e}")
        return "English"

def detect_domain(text):
    """
    根据文本内容识别领域，若检测异常返回默认领域 "信息技术"
    """
    cleaned_text = pattern.sub('', text)
    if not cleaned_text:
        return "信息技术"
    
    try:
        # 构造批次输入数据，文本与各领域池并列传入
        batch = [[cleaned_text, *pools]]
        scores = reranker.compute_score(batch)
        if scores and scores[0]:
            max_score_index = scores[0].index(max(scores[0]))
            domain_name = pools[max_score_index].split(":")[0]
            return domain_name
        else:
            return "信息技术"
    except Exception as e:
        logger.error(f"领域检测错误: {e}")
        return "信息技术"

def process_data(input_path, output_path, limit=None):
    """
    处理数据：加载 JSON 数据、检测语言和领域、保存处理后的文件

    参数:
        input_path (str): 输入文件路径
        output_path (str): 输出文件路径
        limit (int, 可选): 限制处理数据条数
    """
    try:
        with open(input_path, "r") as f:
            data = json.load(f)

        if limit is not None:
            data = data[:limit]
        
        total = len(data)
        logger.info(f"正在处理 {total} 条记录...")

        for item in tqdm(data, total=total):
            conversations = item.get('conversations', [])
            # 当对话记录数量足够时，仅提取中间部分；否则处理全部对话内容
            if len(conversations) >= 3:
                text = "\n".join(j.get('value', '') for j in conversations[1:-2])
            else:
                text = "\n".join(j.get('value', '') for j in conversations)
            
            item['language'] = detect_language(text)
            item['turn_num'] = len(conversations)
            item['domain'] = detect_domain(text)
        
        with open(output_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info("数据处理成功完成！已保存至：" + output_path)
    except Exception as e:
        logger.error(f"处理数据时发生错误: {e}")

if __name__ == "__main__":
    input_json_path = "/root/app/rag_data/qwen_bench/data_clsed/qwen_bench_46809_train.json"
    output_json_path = "/root/app/rag_data/qwen_bench/data_clsed/qwen_bench_46809_train_labelled.json"
    process_data(input_json_path, output_json_path, limit=None)