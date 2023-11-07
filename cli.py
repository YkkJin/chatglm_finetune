#!/root/miniconda3/envs/glm/bin/python
import click
from datetime import date 
from finnlp.data_sources.report.eastmoney_report import Eastmoney_Downloader
from config.config import proxy
from config.constants import DATA_DIR
import os 
from data.preprocessor import FinReportPreprocessor
from data.utils import *
import pandas as pd
import datasets
import torch
from torch.utils.data import DataLoader 
from transformers import  AutoModel,AutoTokenizer,AutoConfig,DataCollatorForSeq2Seq
from peft import get_peft_model, AdaLoraConfig, TaskType,PeftModel
from tqdm import tqdm
from config.config import cfg
from finglm.utils import glm_preprocess,load_model,load_data,load_data_all,create_weighted_sampler
from finglm.train import StepRunner,save_ckpt,load_ckpt
from torchkeras import KerasModel
from datetime import datetime,date
from logging import Logger
import warnings 
warnings.filterwarnings("ignore")


TODAY = date.today()
FIRST_DAY_OF_MONTH = date(TODAY.year,TODAY.month,1)

@click.command()
@click.option("--start-date",default = FIRST_DAY_OF_MONTH.strftime("%Y-%m-%d") )
@click.option("--end-date",default = TODAY.strftime("%Y-%m-%d"))
@click.option("--report-type",default = 0)
@click.option("--download-pdf",default = True)
@click.option("--save-to-path",default = DATA_DIR )
def download(start_date,end_date,report_type,download_pdf,save_to_path):
    report_downloader = Eastmoney_Report(proxy,report_type)
    download_df = report_downloader.download_date_range(start_date=start_date,end_date=end_date,if_download_pdf = download_pdf)
    download_df.to_csv(os.path.join(save_to_path,'report_raw_data',f"{start_date}_{end_date}.csv"),index = False)

@click.command()
@click.option("--year")
@click.option("--label-freq",default = 'week')
def preprocess(year,label_freq):
    processor = FinReportPreprocessor(year)
    processor.parse()
    df = pd.read_csv(os.path.join(DATA_DIR,'report_processed_data',f"{year}.csv"))
    print(DATA_DIR)
    build_prompt(df)
    build_response(df,label_freq)
    #df = df[['prompt','response']]
    if not os.path.exists(os.path.join(DATA_DIR,'report_feed_data')):
        os.makedirs(os.path.join(DATA_DIR,'report_feed_data'))
    df.to_csv(os.path.join(DATA_DIR,'report_feed_data',f"{year}.csv"),index = False)

@click.command()
@click.option("--year")
@click.option("--model-name",default = "chatglm2-6b")
def train(year,model_name):
    year_list = [i for i in range(2017,int(year))]
    data_all = load_data_all(year_list)
    validation_df = load_data(year)
    tokenizer,model = load_model(model_name)
    data_all = data_all[['prompt','response']]
    data_all.dropna(inplace=True)
    train_df = data_all

    # 检查是否有上一年的adapter，如果有则合并lora权重
    #ckpt_path = f'finglm/full_year/{year_list[-2]}' #
    #if os.path.exists(ckpt_path):
        #peft_loaded = PeftModel.from_pretrained(model,ckpt_path).cuda()
        #model = peft_loaded.merge_and_unload() #合并lora权重
    
    
    ds_train_raw = datasets.Dataset.from_pandas(train_df[['prompt','response']])
    ds_val_raw = datasets.Dataset.from_pandas(validation_df[['prompt','response']])
    
    logging_info = f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Starting LoRA finetune task for {year}\n"
    with open('log/train.log','a') as f:
        f.write(logging_info)
        
    ds_train = ds_train_raw.map(
            glm_preprocess,
            batched= True,
            num_proc= 4,
            remove_columns=ds_train_raw.column_names,
            fn_kwargs={'tokenizer':tokenizer,'cfg':cfg}
        )
    ds_val = ds_val_raw.map(
            glm_preprocess,
            batched = True,
            num_proc=4,
            remove_columns= ds_val_raw.column_names,
            fn_kwargs={'tokenizer':tokenizer,'cfg':cfg}
        )
    
    
    data_collator = DataCollatorForSeq2Seq( 
            tokenizer,
            model=None,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            padding=False
        )
    
    sampler = create_weighted_sampler(train_df['response'])
    dl_train = DataLoader(ds_train,batch_size = cfg.batch_size,
                          num_workers = 2, sampler=sampler, collate_fn = data_collator 
                         )
    dl_val = DataLoader(ds_val,batch_size = cfg.batch_size,
                          num_workers = 2, shuffle = False, collate_fn = data_collator 
                         )
        
    
    
    
        #训练时节约GPU占用
    model.config.use_cache=False
    model.supports_gradient_checkpointing = True  #
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM, 
			inference_mode=False,
            r=8,
            lora_alpha=32, 
			lora_dropout=0.1,
            target_modules=["query", "value"]
        )
    
    peft_model = get_peft_model(model, peft_config)
    
    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    peft_model.print_trainable_parameters()
    
    KerasModel.StepRunner = StepRunner 
    KerasModel.save_ckpt = save_ckpt 
    KerasModel.load_ckpt = load_ckpt 
    optimizer = torch.optim.AdamW(peft_model.parameters(),lr=cfg.lr) 
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer,step_size = cfg.scheduler_steps,gamma = cfg.scheduler_gamma)
    keras_model = KerasModel(peft_model,loss_fn =None,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
                ) 
    ckpt_path = f'finglm/full_year/{year}'
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    
    
    keras_model.fit(train_data = dl_train,
                    val_data = dl_val,
                    epochs=cfg.epochs,
                    patience=20,
                    monitor='val_loss',
                    mode='min',
                    ckpt_path = ckpt_path,
                    mixed_precision='fp16',
                    gradient_accumulation_steps = cfg.gradient_accumulation_steps
                   )
        
    del peft_model
    torch.cuda.empty_cache()
    with open('log/train.log','a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} LoRA task done for {year}")
    os.system("shutdown -s -t 10")



@click.command()
@click.option("--year")
@click.option("--month")
@click.option("--model-name",default = "finglm")
def monthly_train(year,month,model_name):
    
    data_all = load_data(year)
    tokenizer,model = load_model(model_name)
    train_df = data_all.query(f'month == {month}')
    train_df = train_df[['prompt','response']]
    train_df.dropna(inplace=True)
    
    validation_df = train_df
	
    

    # 检查是否有上一年的adapter，如果有则合并lora权重
    #ckpt_path = f'finglm/full_year/{year_list[-2]}' #
    #if os.path.exists(ckpt_path):
        #peft_loaded = PeftModel.from_pretrained(model,ckpt_path).cuda()
        #model = peft_loaded.merge_and_unload() #合并lora权重
    
    
    ds_train_raw = datasets.Dataset.from_pandas(train_df[['prompt','response']])
    ds_val_raw = datasets.Dataset.from_pandas(validation_df[['prompt','response']])
    
    logging_info = f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Starting LoRA finetune task for {year}\n"
    with open('log/train.log','a') as f:
        f.write(logging_info)
        
    ds_train = ds_train_raw.map(
            glm_preprocess,
            batched= True,
            num_proc= 4,
            remove_columns=ds_train_raw.column_names,
            fn_kwargs={'tokenizer':tokenizer,'cfg':cfg}
        )
    ds_val = ds_val_raw.map(
            glm_preprocess,
            batched = True,
            num_proc=4,
            remove_columns= ds_val_raw.column_names,
            fn_kwargs={'tokenizer':tokenizer,'cfg':cfg}
        )
    
    
    data_collator = DataCollatorForSeq2Seq( 
            tokenizer,
            model=None,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            padding=False
        )
    sampler = create_weighted_sampler(train_df['response'])

    
    
    dl_train = DataLoader(ds_train,batch_size = cfg.batch_size,
                          num_workers = 2, sampler=sampler, collate_fn = data_collator 
                         )
    dl_val = DataLoader(ds_val,batch_size = cfg.batch_size,
                          num_workers = 2, shuffle = False, collate_fn = data_collator 
                         )
        
    
    
    
        #训练时节约GPU占用
    model.config.use_cache=False
    model.supports_gradient_checkpointing = True  #
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM, 
			inference_mode=False,
            r=8,
            lora_alpha=32, 
			lora_dropout=0.1,
            target_modules=["query", "value"]
        )
    
    peft_model = get_peft_model(model, peft_config)
    
    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    peft_model.print_trainable_parameters()

    KerasModel.StepRunner = StepRunner 
    KerasModel.save_ckpt = save_ckpt 
    KerasModel.load_ckpt = load_ckpt 
    optimizer = torch.optim.AdamW(peft_model.parameters(),lr=cfg.lr) 
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer,step_size = cfg.scheduler_steps,gamma = cfg.scheduler_gamma)
    keras_model = KerasModel(peft_model,loss_fn =None,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
                ) 
    ckpt_path = f'finglm/full_year/{year}'
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    
    
    keras_model.fit(train_data = dl_train,
                    val_data = dl_val,
                    epochs=cfg.epochs,
                    patience=20,
                    monitor='val_loss',
                    mode='min',
                    ckpt_path = ckpt_path,
                    mixed_precision='fp16',
                    gradient_accumulation_steps = cfg.gradient_accumulation_steps
                   )
        
    del peft_model
    torch.cuda.empty_cache()
    with open('log/train.log','a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} LoRA task done for {year}")
    os.system("shutdown -s -t 10")
	
	
	
@click.command()
@click.option('--year')
@click.option('--model-name')
def merge(year,model_name):
    tokenizer,model_base = load_model(model_name)
    ckpt_path = f'finglm/full_year/{year}'
    peft_model = PeftModel.from_pretrained(model_base,ckpt_path).cuda()
    model_new = peft_model.merge_and_unload()
    model_save_path = '/root/autodl-tmp/models/finglm'
    model_new.save_pretrained(model_save_path,max_shard_size = '2GB')


@click.command()
@click.option('--year')
@click.option('--model-name')
def predict(year,model_name):
    data_all = load_data(year)
    tokenizer,model = load_model(model_name)

    def build_response(prompt,model,tokenizer):
        prompt = cfg.source_prefix + prompt 
        lora_response = model.chat(query=prompt , tokenizer = tokenizer)[0]
        return lora_response
    week_list = data_all['week'].unique()
    for week in tqdm(week_list):
        
        output_path = f'data/lora_output_data/full_year/highly_recommend/{year}/{week}'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        test_df = data_all.query(f'year == {year} and week == {week}')
        #print(test_df['week'].unique())
        test_df['lora_response']= test_df['prompt'].apply(build_response,args = (model,tokenizer))
        test_df.to_csv(os.path.join(output_path,'full_result.csv'),index = False) 
        highly_recommend = test_df[test_df['lora_response'].str.contains('强烈推荐')]
        if highly_recommend.empty:
            continue
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
    torch.cuda.empty_cache()

@click.command()
@click.option('--year')
@click.option('--month')
@click.option('--model-name',default='finglm')
def monthly_predict(year,month,model_name):
    data_all = load_data(year)
    data_all = data_all.query(f'month == {month}')
    tokenizer,model = load_model(model_name)
    def build_response(prompt,model,tokenizer):
        prompt = cfg.source_prefix + prompt 
        lora_response = model.chat(query=prompt , tokenizer = tokenizer)[0]
        return lora_response
    week_list = data_all['week'].unique()
    for week in tqdm(week_list):
        
        output_path = f'data/lora_output_data/full_year/highly_recommend/{year}/{week}'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        test_df = data_all.query(f'year == {year} and week == {week}')
        #print(test_df['week'].unique())
        test_df['lora_response']= test_df['prompt'].apply(build_response,args = (model,tokenizer))
        test_df.to_csv(os.path.join(output_path,'full_result.csv'),index = False) 
        highly_recommend = test_df[test_df['lora_response'].str.contains('强烈推荐')]
        if highly_recommend.empty:
            continue
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
    torch.cuda.empty_cache()
	

@click.group()
def cli():
    pass

cli.add_command(download)
cli.add_command(preprocess)
cli.add_command(train)
cli.add_command(merge)
cli.add_command(predict)
cli.add_command(monthly_train)
cli.add_command(monthly_predict)



if __name__ == "__main__":
    cli()