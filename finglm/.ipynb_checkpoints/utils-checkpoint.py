#这是支持 history列处理，并且按照batch预处理数据的方法。
from datasets import Dataset
from config.config import * 
import pandas as pd
import torch
from transformers import  AutoModel,AutoTokenizer,AutoConfig,DataCollatorForSeq2Seq,AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from collections import Counter,defaultdict
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler 
import torch



def load_data(year):
    data = pd.read_csv(f'data/report_feed_data/{year}.csv',index_col = False)
    data['publishDate'] = pd.to_datetime(data['publishDate'])
    data.sort_values(by = 'publishDate',inplace = True)
    return data

def load_data_all(year_list):
    data_all = pd.DataFrame()

    for year in year_list:
        data = pd.read_csv(f'data/report_feed_data/{year}.csv',index_col = False)
        data_all = pd.concat([data_all,data],axis = 0)
    data_all['publishDate'] = pd.to_datetime(data_all['publishDate'])
    data_all.sort_values(by = 'publishDate',inplace = True)
    return data_all

def load_model(model_name):
	config = AutoConfig.from_pretrained(cfg.model_name_or_path[model_name], trust_remote_code=True)
	tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path[model_name], trust_remote_code=True)
	model = AutoModel.from_pretrained(cfg.model_name_or_path[model_name],config=config,trust_remote_code=True).half().cuda()
	return tokenizer,model
def generate_rolling_date_config(data,rolling_window,year_list):
    month_list = [12,1,2,3,4,5,6,7,8,9,10,11] 
    training_start_year = year_list[0]
    training_start_month = data.month.min()

    rolling_date_config = {
    'training_start': {
        'period': [],
    },
    'training_end': {
        'period' :[]
    },
    'validation':{
        'period':[]
    },
    'test':{
        'period':[]
    },
    'rolling times': 0
    }

    training_end_year = training_start_year
    training_end_month = training_start_month + rolling_window

    validation_year = training_end_year 
    validation_month = 1

    test_year = validation_year 
    test_month = validation_month+1

    while test_year <= data.year.max() and test_month <= data.month.max():
        training_end_month = month_list[(training_start_month + rolling_window)%12]
        if (training_start_month + rolling_window)%12 == 1:
            training_end_year += 1
        rolling_date_config['training_start']['period'].append((training_start_year,training_start_month))
        rolling_date_config['training_end']['period'].append((training_end_year,training_end_month))
        validation_month = month_list[(training_end_month + 1)%12] 
        if (training_end_month + 1)%12 == 1:
            validation_year += 1
        rolling_date_config['validation']['period'].append((validation_year,validation_month))
        test_month = month_list[(validation_month+ 1)%12]
        test_year = validation_year
        if (validation_month+ 1)%12 == 1:
            test_year += 1 
        rolling_date_config['test']['period'].append((test_year,test_month))
        if (training_start_month+1)%12 == 1:
            training_start_year += 1
        training_start_month = month_list[(training_start_month+1)%12]
        rolling_date_config['rolling times'] += 1
    return rolling_date_config


def glm_preprocess(examples,tokenizer,cfg):
    max_seq_length = cfg.max_source_length + cfg.max_target_length
    model_inputs = {
        "input_ids": [],
        "labels": [],
    }
    for i in range(len(examples[cfg.prompt_column])):
        if examples[cfg.prompt_column][i] and examples[cfg.response_column][i]:
            query, answer = examples[cfg.prompt_column][i], examples[cfg.response_column][i]

            history = examples[cfg.history_column][i] if cfg.history_column is not None else None
            prompt = tokenizer.build_prompt(query, history)

            prompt = cfg.source_prefix + prompt
            a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                     max_length=cfg.max_source_length)
            b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                     max_length=cfg.max_target_length)

            context_length = len(a_ids)
            input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
            labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
    return model_inputs

#仅仅保存lora相关的可训练参数
def save_ckpt(self, ckpt_path='checkpoint', accelerator = None):
    unwrap_net = accelerator.unwrap_model(self.net)
    unwrap_net.save_pretrained(ckpt_path)
    
def load_ckpt(self, ckpt_path='checkpoint'):
    self.net = self.net.from_pretrained(self.net.base_model.model,ckpt_path)
    self.from_scratch = False

	

def create_weighted_sampler(label):
     classes = label.unique()
     weight_dict = defaultdict()
     class_weight = compute_class_weight('balanced',classes=classes,y = label.values)
     for cls,weight in zip(classes,class_weight):
          weight_dict[cls]=weight
     sampler = WeightedRandomSampler(weights=torch.tensor([weight_dict[cls] for cls in label.values]).to('cuda:0'),num_samples=len(label),replacement=True)
     return sampler


     



     