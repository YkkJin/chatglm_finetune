from datetime import datetime,date,timedelta
import akshare as ak
import os
import pandas as pd
from config.constants import DATA_DIR
from time import sleep
import numpy as np

def symbol_converter(code):
    return code[:-3]

def check_history_price_file(code):
    return os.path.exists(os.path.join(DATA_DIR,'stock_data',code+'.csv'))

def get_valid_stock():
    return ak.stock_info_a_code_name()['code']

def get_last_friday():
    today = date.today()
    days_to_friday = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=days_to_friday)
    return last_friday.strftime("%Y-%m-%d")

def find_nearst_friday(code):
    price_table = pd.read_csv(os.path.join(DATA_DIR,'stock_data',code+'.csv'))
    price_table['日期'] = pd.to_datetime(price_table['日期'])
    max_date = price_table['日期'].max()
    while max_date.weekday()!= 4:
        max_date -= timedelta(days=1)
    return max_date.to_pydatetime().strftime("%Y-%m-%d")


def check_history_price_up_to_date(code):
    if check_history_price_file(code) == False:
        return False
    if find_nearst_friday(code) != get_last_friday():
        return False
    return True

def pull_history_price(code):
    # 拉取后赋权全量历史行情周频数据
    #sleep(np.random.choice(5))
    price_df = ak.stock_zh_a_hist(symbol=code,period = 'weekly',adjust='hfq')
    price_df['代码'] = code
    price_df.to_csv(os.path.join(DATA_DIR,'stock_data',code+'.csv'),index=False)
    del price_df
    

def check_local_history_price_file(code):
    if check_history_price_up_to_date(code) == False:
        pull_history_price(code)
    return pd.read_csv(os.path.join(DATA_DIR,'stock_data',code+'.csv'),dtype={'代码':'str'})

def add_date_features(df):
    if 'date' not in df.columns:
        return False 
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.isocalendar()['week']
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return True

def add_return_series(df):
    if 'close' not in df.columns:
        return False 
    df['ret_after_1_week']  = df['close'].pct_change().shift(-1)
    df['ret_after_1_month'] = df['close'].pct_change(4).shift(-4)
    df['ret_after_1_year']  = df['close'].pct_change(52).shift(-52)
    return True





def build_date_desc(reportDate):
    

    return f"这篇报告的发布日期是：{reportDate}"

def build_stock_desc(row):
    return f"该报告标题为: {row['title']}, 该报告提到的个股为: {row['stockName']}, 从属的行业为: {row['indvInduName']}"

def build_report_meta(row):
    row['report_meta'] = row['data_desc']+'\n' + row['author_desc'] + '\n' + row['stock_desc']

def build_author_desc(row):
    return f"这篇报告的作者是：{row['researcher']},所属机构为 {row['orgSName']}, 作者给出的评级为: {row['sRatingName']}"


def build_ret_label(ret_after_1_month):
    if pd.isnull(ret_after_1_month) == True:
        return
    if ret_after_1_month < -0.05:
        return "强烈不推荐"
    elif    ret_after_1_month < -0.02 and ret_after_1_month >= -0.05:
        return "不推荐"
    elif ret_after_1_month >= -0.02 and ret_after_1_month <= 0.02:
        return "一般"
    elif ret_after_1_month > 0.02 and ret_after_1_month <=0.05:
        return "推荐"
    else: #收益率大于5%
        return "强烈推荐"


def build_prompt(df):

    df['date_desc'] = df['publishDate'].apply(build_date_desc)

    df['author_desc'] =  df.apply(build_author_desc,axis=1)
    
    df['stock_desc'] = df.apply(build_stock_desc,axis = 1)

    df['prompt'] = df['date_desc']+'\n' + df['author_desc'] + '\n' + df['stock_desc']
    return df 

def build_response(df):
    df['response'] = df['ret_after_1_month'].apply(build_ret_label)
    return df