import pandas as pd 
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
import calendar
from finglm.utils import *
from logging import Logger

year_list = [2021,2022,2023]

data_all = load_data_all(year_list)

rolling_date_config = generate_rolling_date_config(data=data_all,rolling_window=5,year_list = year_list)



config = AutoConfig.from_pretrained(cfg.model_name_or_path['chatglm2-6b'], trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path['chatglm2-6b'], trust_remote_code=True)
model = AutoModel.from_pretrained(cfg.model_name_or_path['chatglm2-6b'],config=config,trust_remote_code=True).half().cuda()


for idx in range(17,rolling_date_config['rolling times']):
    start_year,start_month = rolling_date_config['training_start']['period'][idx]
    end_year,end_month = rolling_date_config['training_end']['period'][idx]
    validation_year,validation_month = rolling_date_config['validation']['period'][idx]
    test_year,test_month = rolling_date_config['test']['period'][idx]


    start_date = date(start_year,start_month,1).strftime("%Y-%m-%d")
    _,lastday = calendar.monthrange(end_year,end_month)
    end_date = date(end_year,end_month,lastday).strftime("%Y-%m-%d")
    train_df = data_all[(data_all['publishDate']>=start_date) &  (data_all['publishDate']<=end_date)]
    validation_df = data_all.query(f'year == {validation_year} and month == {validation_month}')
    test_df = data_all.query(f'year == {test_year} and month == {test_month}')
    ds_train_raw = datasets.Dataset.from_pandas(train_df[['prompt','response']])
    ds_val_raw = datasets.Dataset.from_pandas(validation_df[['prompt','response']])
    ds_test_raw = datasets.Dataset.from_pandas(test_df[['prompt','response']])

    logging_info = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Starting LoRA finetune task for {test_year}/{test_month}\n"
    with open('log/train.log','a') as f:
        f.write(logging_info)
    print(logging_info)
    
    ds_train = ds_train_raw.map(
        preprocess,
        batched= True,
        num_proc= 4,
        remove_columns=ds_train_raw.column_names,
        fn_kwargs={'tokenizer':tokenizer,'cfg':cfg}
    )
    ds_val = ds_val_raw.map(
        preprocess,
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


    dl_train = DataLoader(ds_train,batch_size = cfg.batch_size,
                      num_workers = 2, shuffle = True, collate_fn = data_collator 
                     )
    dl_val = DataLoader(ds_val,batch_size = cfg.batch_size,
                      num_workers = 2, shuffle = False, collate_fn = data_collator 
                     )
    

    from peft import get_peft_model, AdaLoraConfig, TaskType,PeftModel

    #训练时节约GPU占用
    model.config.use_cache=False
    model.supports_gradient_checkpointing = True  #
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=100,
        lora_alpha=32, lora_dropout=0.1,
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
    ckpt_path = f'finglm/six_month_rolling/{test_year}_{test_month}'
    import os
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
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} LoRA task done for {test_year}/{test_month}\n "  )



    
    