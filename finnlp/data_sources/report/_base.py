from finnlp.data_sources._base import FinNLP_Downloader

class Report_Downloader(FinNLP_Downloader):
    """
    Download the reports 
    """

    def __init__(self,args):
        super().__init__(args)
        #self._report_url = f"https://reportapi.eastmoney.com/report/list?cb=datatable9774938&industryCode=*&pageSize=500&beginTime={start_date}&endTime={end_date}&pageNo={page_no}&qType=0"
        
    
    def download_date_range(self):
        pass


        
    def download_pdf(self):
        pass
    




