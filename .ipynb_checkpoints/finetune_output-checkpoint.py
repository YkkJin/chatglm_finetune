
from transformers import AutoModel,AutoTokenizer 
from peft import PeftModel
import datasets
import torch
from torch.utils.data import DataLoader 
from transformers import  AutoModel,AutoTokenizer,AutoConfig,DataCollatorForSeq2Seq
from config.config import cfg
from finglm.utils import preprocess
from finglm.train import StepRunner 
from torchkeras import KerasModel 
from accelerate import Accelerator
from datetime import datetime,date
from finglm.utils import *
from logging import Logger
import numpy as np
import warnings 
warnings.filterwarnings("ignore")



year_list = [2021,2022,2023]

data_all = load_data_all(year_list)

rolling_date_config = generate_rolling_date_config(data=data_all,rolling_window=5,year_list = year_list)
model_old = AutoModel.from_pretrained(cfg.model_name_or_path['chatglm2-6b'],
                                  load_in_8bit=False, 
                                  trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path['chatglm2-6b'],
                                    trust_remote_code = True)

def build_response(prompt,model,tokenizer):
    prompt = cfg.source_prefix + prompt 
    lora_response = model.chat(query=prompt , tokenizer = tokenizer)[0]
	print(lora_response)
    return lora_response



for idx in range(2,len(rolling_date_config['test']['period'])):
    test_year,test_month = rolling_date_config['test']['period'][idx]
    with open('log/lora_out.log','a') as f:
        f.writelines(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} LoRA merged model start for {test_year}/{test_month}\n" )
   

    ckpt_path = f'finglm/six_month_rolling/{test_year}_{test_month}'
    
    output_path = f'data/lora_output_data/six_month_rolling/highly_recommend/{test_year}_{test_month}'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    test_df = data_all.query(f'year == {test_year} and month == {test_month}')

    peft_loaded = PeftModel.from_pretrained(model_old,ckpt_path).cuda()
    model_new = peft_loaded.merge_and_unload() #合并lora权重
    
    test_df['lora_response']= test_df['prompt'].apply(build_response,args = (model_new,tokenizer))
    
    test_df.to_csv(os.path.join(output_path,'full_result.csv'),index = False) 
    highly_recommend = test_df[test_df['lora_response'].str.contains('强烈推荐')]
    highly_recommend['type'] = 'LoRA'
    highly_recommend['sell_week'] = (highly_recommend['week']+4)%12

    np.random.seed(1)
    random_idx = np.random.choice(len(test_df),10)
    recommend_random_idx = np.random.choice(len(highly_recommend),10)
    highly_recommend = highly_recommend.iloc[recommend_random_idx]
    benchmark = test_df.iloc[random_idx]
    benchmark['type'] = 'random'
    benchmark['sell_week'] = (benchmark['week']+4)%12
    
    result_df = pd.concat([highly_recommend,benchmark])
    selected_cols = ["publishDate","stockName","stockCode","week","month","year","sell_week","type"]
    result_df = result_df[selected_cols]

    result_df.to_csv(os.path.join(output_path,'choose_result.csv'),index = False)
    del peft_loaded
    torch.cuda.empty_cache()


