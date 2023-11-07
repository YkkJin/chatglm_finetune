import pandas as pd
import os 
import ast
from tqdm import tqdm
from datetime import datetime
from data.utils import *



class FinReportPreprocessor():
    def __init__(self,year):
        self.year = year
        self.raw_report_data = pd.read_csv(os.path.join(DATA_DIR,'report_raw_data',f'{self.year}.csv'),dtype='str')
        print(f'Init PreProcessor, Original data contains {len(self.raw_report_data)} reports')

    def _parse_date(self):
        # 具体实现解析日期的逻辑
        self.raw_report_data['publishDate'] = pd.to_datetime(self.raw_report_data['publishDate'])
        self.raw_report_data['date'] = self.raw_report_data['publishDate'].dt.date
        add_date_features(self.raw_report_data)
        
        

    def _parse_delist_stock(self):
        valid_stock_df = get_valid_stock().to_list()
        self.raw_report_data = self.raw_report_data[self.raw_report_data['stockCode'].isin(valid_stock_df)]
        
        print(f'Removing delisted stock, Remaining {len(self.raw_report_data)} reports')


    def _parse_orgprizeinfo(self):
        # 具体实现解析组织奖项信息的逻辑
        pass
            
        
        
    
    def _get_stock_price(self):

        # 获取 _raw_report_data 中包含的股票代码集合
        stock_codes = self.raw_report_data['stockCode'].unique().tolist()

        # 获取股票最小报告日期，以确定有效的股票价格查询时间范围
        # stock_min_date = self.raw_report_data.loc[self.raw_report_data.groupby('code')['reportDate'].idxmin()][['code','reportDate']]
        # stock_max_date = self.raw_report_data.loc[self.raw_report_data.groupby('code')['reportDate'].idxmax()][['code','reportDate']]
        # 遍历股票代码，获取股价数据
        stock_price = pd.DataFrame({'date':[],'close':[],'ret_after_1_week':[],'ret_after_1_month':[],'ret_after_1_year':[]})
        for code in tqdm(stock_codes,desc="Merging Price Table"):

            columns_map = {
                '日期':'date',
                '收盘':'close',
                '代码':'stockCode'
            }
            # 获取当前股票代码历史行情收盘价格(周频，后复权)

            current_stock_weekly_price = check_local_history_price_file(code).rename(columns=columns_map)
            current_stock_weekly_price = current_stock_weekly_price[['date','close','stockCode']]
            # 构造week/month/year特征用于join
            add_date_features(current_stock_weekly_price)
            # 构造收益率序列（周/月/年）
            add_return_series(current_stock_weekly_price)
            stock_price = pd.concat([stock_price,current_stock_weekly_price],axis=0)
        
        stock_price['stockCode'] = stock_price['stockCode'].astype('str')
        stock_price['year'] = stock_price['year'].astype('int64')
        self.raw_report_data['year'] = self.raw_report_data['year'].astype('int64')
        stock_price['month'] = stock_price['month'].astype('int64')
        self.raw_report_data['month'] = self.raw_report_data['month'].astype('int64')
        stock_price['week'] = stock_price['week'].astype('int64')
        self.raw_report_data['week'] = self.raw_report_data['week'].astype('int64')
        assert(stock_price['stockCode'].dtype == self.raw_report_data['stockCode'].dtype)
        assert(stock_price['year'].dtype == self.raw_report_data['year'].dtype)
        assert(stock_price['month'].dtype == self.raw_report_data['month'].dtype)
        assert(stock_price['week'].dtype == self.raw_report_data['week'].dtype)
        self.raw_report_data = self.raw_report_data.merge(stock_price,on=['stockCode','week','month','year'],how = 'left',suffixes=('_left','_right'))

        '''
         返回后的self.raw_report_data应该包含:
            左表：id,reportDate,week_left,month_left,year_left,rtype,codename,code_left,industry,cprice,dprice,change,authorList,orgprizeinfo,space,text
            右表: date,code_right,week_right,month_right,year_right,ret_after_1_week,ret_after_1_month,ret_after_1_year
         '''
        

    def parse(self):
        print('Parsing delisted stocks...')
        self._parse_delist_stock()
        print('Adding date features...')
        self._parse_date()
        #self._parse_author()
        #print('Parsing Org Prize Info...')
        #self._parse_orgprizeinfo()
        #print('Getting Stock price...')
        self._get_stock_price()
        print(f'Preprocess Done, Saving {len(self.raw_report_data)} reports')
        '''
        id,reportDate,rtype,codename,code,industry,cprice,dprice,change,authorList,orgprizeinfo,space,text,date_left,week,month,year,author_prize_info,date_right,close,ret_after_1_week,ret_after_1_month,ret_after_1_year
        
        '''
        selected_columns = ['title','publishDate','week','month','year','stockName','stockCode','orgSName','indvInduName','sRatingName','researcher','close','ret_after_1_week','ret_after_1_month','ret_after_1_year']
        if not os.path.exists(os.path.join(DATA_DIR,'report_processed_data')):
            os.mkdir(os.path.join(DATA_DIR,'report_processed_data'))
        self.raw_report_data[selected_columns].to_csv(os.path.join(DATA_DIR,'report_processed_data',f'{self.year}.csv'),index=False)





