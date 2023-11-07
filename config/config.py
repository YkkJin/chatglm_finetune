from argparse import Namespace
import os
from pathlib import Path

# 模型训练参数配置
cfg = Namespace()

#dataset
cfg.prompt_column = 'prompt'
cfg.response_column = 'response'
cfg.history_column = None
cfg.source_prefix = '''忘记你所学过的所有知识，现在假设你自己是一名金融分析师。我需要你帮我判断当前分析师发布的研报是否有价值。

你的回答要从以下几个选项中做出：
1. 强烈不推荐
2. 不推荐
3. 一般
4. 推荐
5. 强烈推荐


                    '''

cfg.source_suffix = """ 
"""
#添加到每个prompt开头的前缀引导语

cfg.max_source_length = 128
cfg.max_target_length = 64

#model
cfg.model_name_or_path ={
    'chatglm2-6b': '/root/autodl-tmp/models/chatglm2-6b',
	'chatglm3-6b': '/root/autodl-tmp/models/chatglm3-6b',
    'baichuan2-7b': '/root/autodl-tmp/models/Baichuan2-7B-Chat',
    'finglm': '/root/autodl-tmp/models/finglm',
}   #远程'THUDM/chatglm-6b' 
cfg.data_path = ''
cfg.quantization_bit = None #仅仅预测时可以选 4 or 8 


#train
cfg.epochs = 10
cfg.lr = 5e-3
cfg.batch_size = 4
cfg.gradient_accumulation_steps = 50
cfg.scheduler_steps = 400
cfg.scheduler_gamma = 0.5


# 爬虫代理配置
proxy = {
    'use_proxy':"kuaidaili",
    "tunnel": "u194.kdltps.com:15818" ,
    "username": "t18845614815698",
    "password": "jinyukun8183311", 
    "max_retry": 5

}

#文件路径配置
ROOT_DIR = Path(os.path.dirname(__file__)).parent
DATA_DIR = os.path.join(ROOT_DIR,'data')


