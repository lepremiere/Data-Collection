import os 
import datetime
import requests as r
import pandas as pd
import polars as pl
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from pandas.tseries.offsets import BDay
from lib.baseClass import BaseClass

class EODDownloader(BaseClass):

    def __init__(self, api_key, exchanges, folder=None, verbose=True):
        super().__init__(folder=folder, verbose=verbose)
        self.host = "https://eodhistoricaldata.com/api/"
        self.api_key = api_key
        self.exchanges = exchanges
        self.check_ticker()

    def check_ticker(self):
        self.print(msg=f"Checking Ticker Lists...")
        folders = ["Temp", "TickerInfo", "TickerInfo/TickerLists", "Data"]
        for folder in folders:
            self.create_folder(folder)

        if not os.path.isfile(f"{self.path}/TickerInfo/exchangesList.csv"):
            self.print(msg=f"Getting available exchanges...")
            url = "exchanges-list/?"
            exchanges_list = pd.DataFrame.from_dict(self.get_response(url=url))
            exchanges_list.to_csv(f"{self.path}/TickerInfo/exchangesList.csv")
        
        exchanges_list = pd.read_csv(f"{self.path}/TickerInfo/exchangesList.csv")
        for name, exchange in exchanges_list.loc[:, ["Name", "Code"]].to_numpy():
            if not os.path.isfile(f"{self.path}/TickerInfo/TickerLists/{exchange}.csv"):
                try:
                    if exchange == "MONEY" or exchange == "BOND":
                        break
                    self.print(msg=f"Getting available ticker for {name} - {exchange}...")
                    url = f"exchange-symbol-list/{exchange}?"
                    ticker_lsit = pd.DataFrame.from_dict(self.get_response(url=url))
                    ticker_lsit.to_csv(f"{self.path}/TickerInfo/TickerLists/{exchange}.csv")
                except Exception as e:
                    self.print(msg=f"Exchange: {name} - {exchange} failed due to: {e}")

    def get_available_files(self):

        self.print(msg=f"Getting available files!")
        folder = f"{self.path}/Data/"
        d = {}
        for root, dirs, files in os.walk(folder):
            d[root] = files

        hits = []
        for key, files in d.items():
            if len(files) > 0 and files:
                for file in tqdm(files):
                    symbol = str(file.split("_")[0])
                    typ, t = key.split("/")[-1].split("\\")
                    path = f"{key}/{file}"
                    df = pl.read_csv(path).to_pandas()
                    start, end = df.date.iloc[[0, -1]]
                    start = start.strftime("%Y-%m-%d")
                    end = end.strftime("%Y-%m-%d")
                    hits.append([symbol, typ, t, start, end, path])
        if len(hits) > 0:
            hits = pd.DataFrame(hits, columns=["symbol", "type", "timeframe", "start", "end", "path"], index=None)
            hits.reset_index(inplace=True)
            hits.drop(columns="index", inplace=True)
            pl.from_pandas(hits).to_csv(f"{self.path}/Temp/available_files.csv")
            self.get_infos()
            self.print(msg=f"{len(hits)} files found!")
        else:
            self.print(msg=f"No available files found!")
    
    def get_infos(self):

        self.print(msg=f"Getting according infos!")
        if not os.path.isfile(f"{self.path}/Temp/available_files.csv"):
            self.get_available_files()
        df = pl.read_csv(f"{self.path}/Temp/available_files.csv").to_pandas()
        df.set_index("symbol", inplace=True)
        df["name"] = np.nan
        df["isin"] = np.nan

        infos = pd.concat([pl.read_csv(f"{self.path}/TickerInfo/TickerLists/{exchange}.csv").to_pandas()\
                                             for exchange in self.exchanges])                
        infos.set_index('Code', inplace=True)
        duplicates = pd.read_csv(f"{self.path}/duplicates.csv")

        for i in tqdm(range(len(df))):    
            symbol = df.index[i]
            type = df.type[i]

            if symbol in duplicates.Code.values:
                ind = np.logical_and(duplicates.Code == symbol, duplicates.Type == type)
                print(duplicates.Name.loc[ind], symbol)
                name = duplicates.Name.loc[ind].values[0]
                isin = duplicates.Isin.loc[ind].values[0]
                exchange = duplicates.Exchange.loc[ind].values[0]
            else:
                ind = np.logical_and(infos.index == symbol, infos.Type == type)
                temp_df = infos.loc[ind,:]

                if len(np.shape(temp_df)) == 2:
                    name = temp_df.Name[np.argmax([len(s) for s in temp_df.Name])]
                    isin = temp_df.Isin[np.argmax([len(s) for s in temp_df.Isin])]
                    exchange = temp_df.Exchange[0]
                else:
                    name = temp_df.Name
                    isin = temp_df.Isin
                    exchange = temp_df.Exchange

            loc = np.logical_and(df.index == symbol, df.type == type)
            df.loc[loc, "isin"] = isin
            df.loc[loc, "name"] = name
            df.loc[loc, "exchange"] = exchange

        df.reset_index(inplace=True)
        df = df.loc[:, ["symbol", "exchange", "type", "timeframe", "name", "isin", "start", "end", "path"]]
        pl.from_pandas(df).to_csv(f"{self.path}/Temp/available_files.csv")

    def get_response(self, url):
        request = f"{self.host}{url}api_token={self.api_key}&fmt=json"
        requestResponse = r.get(request)
        return requestResponse.json()   
    
    def download_ticker(self, ticker, exchange, timeframe, startdate=None, enddate=None):
        if timeframe == 1440:
            url = f"eod/{ticker}.{exchange}?"
            if startdate:
                url = url + f"from={startdate}&"
            if enddate:
                url = url + f"to={enddate}&"
            df = pd.DataFrame.from_dict(self.get_response(url))
        else:
            if not startdate:
                startdate = pd.to_datetime("2020-10-01 00:00:00")
                enddate = pd.to_datetime(datetime.datetime.now())
            intervals = pd.date_range(start=startdate, end=enddate, freq=f"100D")
            intervals = np.unique(np.concatenate([[startdate], intervals.to_list(), [enddate]]))
            intervals = (pd.to_datetime(intervals) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            df = pd.DataFrame()

            for i in range(1, len(intervals)):
                start = intervals[i-1] 
                end = intervals[i]
                url = f"intraday/{ticker}.{exchange}?from={start}&to={end}&interval={timeframe}m&"
                temp_df = pd.DataFrame.from_dict(self.get_response(url))
                df = pd.concat([df, temp_df])  
        df.drop_duplicates(subset=['date'], inplace=True)

        return df

    def _populate_one(self, ticker, exchange, timeframes, t, types):
        if ticker in self.duplicates.Code.values:
                info = self.duplicates.loc[np.logical_and(self.duplicates.Code == ticker, self.duplicates.Exchange == exchange)]
                if info.Type.values[0] == "None":
                    self.print(msg=f"{t}: {exchange} - {ticker} not downloaded - DUPLICATE.")
        else:
            for timeframe in timeframes:
                if not types or t in types:
                    try:
                        if not os.path.isfile(f"{self.path}/Data/{t}/{timeframe}/{ticker}_{timeframe}.csv"):
                            df = self.download_ticker(ticker=ticker, exchange=exchange, timeframe=timeframe)
                            pl.from_pandas(df).to_csv(f"{self.path}/Data/{t}/{timeframe}/{ticker}_{timeframe}.csv")
                            self.print(msg=f"{t}: {exchange} - {ticker}_{timeframe}\tfrom {pd.to_datetime(df.date.iloc[1]).date()} till {pd.to_datetime(df.date.iloc[-1]).date()} successfully downloaded!")
                        else:
                            self.print(msg=f"{t}: {exchange} - {ticker}_{timeframe}\talready exists!")
                    except Exception as e:
                        self.print(msg=f"{t}: {exchange} - {ticker}_{timeframe} failed due to: {e}")

    def populate_tickers(self, types, timeframes):
        if self.verbose:
            print("###################################################\n\
                  Populating Tickers!\n###################################################")

        for typ in types:
            self.create_folder(f"Data/{typ}")
            for timeframe in timeframes:
                self.create_folder(f"Data/{typ}/{timeframe}")

        ticker_list = pd.concat([pl.read_csv(f"{self.path}/TickerInfo/TickerLists/{exchange}.csv").to_pandas()\
                                    for exchange in self.exchanges])
        self.duplicates = pd.read_csv("D:/EOD/duplicates.csv")

        if not types:
            for typ in ticker_list.Type.unique():
                self.create_folder(f"Data/{typ}") 
                for timeframe in timeframes:
                    self.create_folder(f"Data/{typ}/{timeframe}")

            l = []
            for code, name, exchange, t, isin in ticker_list.loc[:, ["Code", "Name", "Exchange", "Type", "Isin"]].to_numpy():
                l.append((code, exchange, timeframes, t, types))
            with Pool(36) as p:
                p.starmap(self._populate_one, l)
                                      
        self.get_available_files()

    def update_files(self):

        self.print(msg=f"Updating files!")
        if not os.path.isfile(f"{self.path}/Temp/available_files.csv"):
            self.get_available_files()
        df = pl.read_csv(f"{self.path}/Temp/available_files.csv").to_pandas()
        iter = df.loc[:,["symbol", "exchange", "type", "timeframe", "start", "end", "path"]].to_numpy()
        df.set_index("symbol", inplace=True)

        for symbol, exchange, type, timeframe, start, end, path\
                in tqdm(iter): 
            try:
                file = pl.read_csv(path).to_pandas()
                update = self.download_ticker(ticker=symbol,
                                            exchange=exchange,
                                            timeframe=timeframe,
                                            startdate=end)
                update = pd.concat([file, update])
                update.date = pd.to_datetime(update.date)
                update.drop_duplicates(subset=['date'], inplace=True)  
                update.date = update.date.apply(lambda x: x.strftime('%Y-%m-%d'))
            except:
                update = self.download_ticker(ticker=symbol,
                                            exchange=exchange,
                                            timeframe=timeframe)
            if update.empty:
                print(symbol, exchange, timeframe, start, end, path, update)
                break

            df.loc[symbol, "end"] = np.datetime64(update.date.iloc[-1])
            pl.from_pandas(update).to_csv(path)   
            # self.print(f"{type}:\t{symbol}_{timeframe} updated from {last} to {update.date.iloc[-1]}!")
        pl.from_pandas(df.reset_index()).to_csv(f"{self.path}/Temp/available_files.csv")  
        self.get_available_files()               
    
if __name__ == "__main__":
    
    api_key = "60abb74aa4a375.36407570"
    folder = "D:/EOD"
    exchanges = ["F", "XETRA"]
    types = []
    timeframes = [1440]
    eod = EODDownloader(api_key=api_key, exchanges=exchanges, folder=folder)
    eod.populate_tickers(types=types, timeframes=timeframes)
    # eod.update_files()
    # eod.get_available_files()

   
# 
