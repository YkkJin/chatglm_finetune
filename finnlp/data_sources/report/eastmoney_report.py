from finnlp.data_sources.report._base import Report_Downloader
from requests import request
import json
import os
from tqdm.auto import tqdm
import pandas as pd
import time
import threading
from config.config import DATA_DIR,proxy


class Eastmoney_Downloader(Report_Downloader):
    """
    Download the reports 
    """

    def __init__(self,args,report_type):
        super().__init__(args)
        self._max_page = None
        self.start_date = None
        self.end_date = None
        self.page_no = 1
        self.report_type = report_type
        #self._report_url = f"https://reportapi.eastmoney.com/report/list?cb=datatable9774938&industryCode=*&pageSize=500&beginTime={start_date}&endTime={end_date}&pageNo={page_no}&qType=0"
        
    
    def download_date_range(self, start_date, end_date, if_download_pdf = True):
        self.start_date = start_date
        self.end_date = end_date
        res_json = self.request_em_report()
        self._max_page = res_json["TotalPage"]
        dir_path = os.path.join(DATA_DIR,'report_raw_data')

        for page_no in tqdm(range(self._max_page)):
            self.page_no = page_no
            res_json = self.request_em_report()
            report_meta_data = res_json["data"]
            for idx,data in enumerate(report_meta_data):
                try:
                    df = pd.DataFrame.from_dict(data).iloc[[0]] # 因为author字段在字典中是一个list，from_dict会自动拆出两条数据，这里只保存第一条数据
                    selected_data = df[["title","infoCode","stockName","stockCode","orgSName","publishDate","indvInduName","emRatingName","lastEmRatingName","researcher","attachPages","sRatingName"]]
                    year = pd.to_datetime(selected_data['publishDate']).dt.year.iloc[0]

                    if not os.path.exists(os.path.join(dir_path,f'{year}.csv')):
                        selected_data.to_csv(os.path.join(dir_path,f'{year}.csv'),mode='w',index=False,header=True)
                    else:
                        selected_data.to_csv(os.path.join(dir_path,f'{year}.csv'),mode='a',index=False,header=False)
                except:
                    continue
            #selected_report_meta_data = report_meta_data["title","infoCode","stockName","stockCode","orgSName","publishDate","indvInduName","emRatingName","lastEmRatingName","researcher","attachPages","sRatingName"]
        if if_download_pdf:
            group_size = len(report_meta_data)//5
            grouped_reports = [report_meta_data[i:i+group_size] for i in range(0,len(report_meta_data),group_size)]
            threads = []
            for i in range(len(grouped_reports)):
                threads.append(threading.Thread(target=self.download_pdf_list,args=(grouped_reports[i],)))
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            
            
            #for report_meta in tqdm(report_meta_data):
                #self.download_pdf(report_meta)
    
    def download_pdf(self, report_meta):
        report_ap_code = report_meta["infoCode"]
        report_title = report_meta["title"].replace("/","")
        report_org_name = report_meta["orgSName"]
        report_publish_date = report_meta["publishDate"]
        date = report_publish_date.split(" ")[0]
        year,month,day = date.split("-")
        save_path = os.path.join(DATA_DIR,'report_pdf',year,month,report_org_name)
        report_name = os.path.join(save_path,report_title+'.pdf')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if os.path.exists(report_name):
            return
        self._download_url = f"https://pdf.dfcfw.com/pdf/H3_{report_ap_code}_1.pdf?1695995282000.pdf"
        res = self._request_get(url = self._download_url)

        if not res:
            return
        with open(report_name,'wb') as f:
            f.write(res.content)
    
    def download_pdf_list(self,report_list):
        for report in tqdm(report_list):
            self.download_pdf(report)
            time.sleep(0.05)

    def request_em_report(self):
        self._report_url = f"https://reportapi.eastmoney.com/report/list?cb=datatable9774938&industryCode=*&pageSize=500&beginTime={self.start_date}&endTime={self.end_date}&pageNo={self.page_no}&qType={self.report_type}"
        res = self._request_get(url = self._report_url)
        if not res:
            return
        res = res.text[17:-1]
        res_json = json.loads(res)
        return res_json





if __name__ == "__main__":

    report_downloader = Report_Donwloader(proxy)
    report_downloader.download_date_range("2023-10-11","2023-10-11")