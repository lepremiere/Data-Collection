import os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import polars as pl
from requests import get
from time import sleep
from datetime import datetime
from multiprocessing import Pool, cpu_count
from lib.baseClass import BaseClass

class TiingoDownloader(BaseClass):
    """ This class provides a solution to automatically 
        downloading price data from API Tiingo. """

    def __init__(self, api_key: str, folder=None, verbose=True):
        """ Class Initialization"""

        super().__init__(folder=folder, verbose=verbose)
        self._print(msg="Initialization!")
        self.verbose = verbose
        self.api_key = api_key
        self.cores = cpu_count()
        self.host = "https://api.tiingo.com/"
        self.headers = {'Content-Type': 'application/json'}
        self.check_ticker()

    def _check_last_update(self, force_update):
        # Check last update to reduce API data load
        if not force_update:
            if os.path.isfile(f"{self.path}/Temp/last_update.txt"):
                today = pd.to_datetime(datetime.now().date())

                with open(f"{self.path}/Temp/last_update.txt", "r") as f:
                    last_update = pd.to_datetime(f.read())
 
                if last_update <= today:
                    self._print(msg=f"Ticker info up to date with last update: {last_update.date()}")
                    s_tickers = pl.read_csv(f"{self.path}/TickerInfo/supported_tickers.csv").to_pandas()
                    s_cryptos = pl.read_csv(f"{self.path}/TickerInfo/supported_crypto.csv").to_pandas()

                    return False, s_tickers, s_cryptos
            else:
                return True, None, None

    def check_ticker(self, force_update=False):
        """ Gets up to date ticker information from API and
            saves results to 'self.path/TickerInfo/' """

        # Check last update to reduce API data load
        flag, s_tickers, s_cryptos = self._check_last_update(force_update=force_update)
        if not flag:
            return s_tickers, s_cryptos

        # creating necessary folders in specified path
        folders = ["Temp", "TickerInfo", "Data"]
        for folder in folders: 
            self._create_folder(folder)

        # Getting list of all supported STOCK, ETF, and FUND ticker 
        self._print(msg=f"Getting available tickers...")
        s_tickers = pd.read_csv(
            "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
            )
        pl.from_pandas(s_tickers).to_csv(f"{self.path}/TickerInfo/supported_tickers.csv")
        self._print(msg=f"Available tickers: {len(s_tickers)}")

        # Getting list of all supported CRYPTO pairs  
        self._print(msg=f"Getting available cryptos...")
        response = self._get_response("/tiingo/crypto?")
        s_cryptos = pd.DataFrame.from_dict(response).drop('description', axis=1)
        s_cryptos = self._reset_idx(s_cryptos)
        s_cryptos = s_cryptos.loc[:, ('ticker', 'name', 'baseCurrency', 'quoteCurrency')]
        s_cryptos = s_cryptos.sort_values('ticker')
        pl.from_pandas(s_cryptos).to_csv(f"{self.path}/TickerInfo/supported_crypto.csv")
        self._print(msg=f"Available cryptos: {len(s_cryptos)}")   

        with open(f"{self.path}/Temp/last_update.txt", "w") as f:
            f.write(str(datetime.now().date()))
            f.close()

        return s_tickers, s_cryptos

    def get_available_files(self):
        """ Screening for available files
        
        This method walks through every subdirectory found in 'self.path', gathers 
        file information and returns it for all available files. Saves results to
        'self.path/Temp/'.

        Args:
        
        Returns:
            Pandas DataFrame: Table containing (ticker, symbol, timeframe, path) infos
        """

        # Walking through subdirectories to detect files
        folder = f"{self.path}/Data/"
        d = {}
        for root, _, files in os.walk(folder):
            d[root] = files

        # Walk through files. Getting attributes
        hits = []
        for key, files in d.items():
            if len(files) > 0:
                for file in files:
                    symbol = str(file.split("_")[0])
                    t = key.split("\\")[1] 
                    typ = key.split("/")[-1].split("\\")[0]
                    path = f"{key}/{file}"
                    hits.append([typ, symbol, t, path])

        hits = pd.DataFrame(hits, columns=["type", "symbol", "timeframe", "path"])

        if hits.empty:
            self._print("No files found")
            return None

        self._print(msg=f"{len(hits)} files found!")
        pl.from_pandas(hits).to_csv(f"{self.path}/Temp/available_files.csv")
        
        return hits
  
    def download_ticker(self, ticker: str, timeframe: int, startdate: str, enddate: str): 
        """ This function downloads a ticker for a given period and sampling frequency.

        Parameters
        ----------
            ticker : str
                Specifies the symbol to download
            timeframe : str
                Specifies the bar frequency. In minutes
            startdate :str
                Date that specifies the first date of interest, fmt: 'YYYY-mm-dd HH:MM:ss'
            enddate : str
                Date that specifies the last date of interest, fmt: 'YYYY-mm-dd HH:MM:ss'
        
        Returns
        -------
        df : pandas.DataFrame
                Table containing dates and price data (ohlcv+)
        """
        # Switching between API endpoints
        if timeframe == 1440:
            df = self._get_eod(ticker=ticker, startdate=startdate)
        else:
            # Defining start and end of period of interest
            startdate, enddate = pd.to_datetime(startdate), pd.to_datetime(enddate)
            span = (pd.to_datetime("2017-01-01 00:00:00"), enddate)
    
            if startdate > span[0]:
                span = (startdate, enddate)
            if span[0] > span[1]:
                return None

            # Creating daterange for the period and timeframe of interest
            # Cutting away dates that are commonly not traded to reduce data usage in API
            intervals = pd.date_range(start=span[0], end=span[1], freq=f"{timeframe}min")
            h = np.array([intervals.hour < 22, intervals.hour > 6, intervals.weekday < 6]).T
            intervals = intervals[h.sum(axis=1) == 3]

            n = 10000               # Max number of rows of Tiingo API requests
            d = len(intervals)//n   # Number of iterations for the loop

            # Batched download if daterange is longer than n
            if len(intervals) > n:
                l = []
                for i in range(n, d*n+1, n):
                    l.append(self._get_iex(ticker, intervals[i-n], intervals[i], timeframe))    
                # Getting rows that are cut off by modulus (d)
                l.append(self._get_iex(ticker=ticker,
                                        startdate=str(pd.to_datetime(l[-1].date.iloc[-1])), 
                                        enddate=str(enddate), 
                                        timeframe=timeframe))
                df = pd.concat(l) 
            else:
                df = self._get_iex(ticker=ticker, 
                                   startdate=str(startdate), 
                                   enddate=str(enddate), 
                                   timeframe=timeframe)
        if type(df) == type(None):
            return None

        df.drop_duplicates(subset=['date'], inplace=True)
        df = df.reset_index().drop(columns="index")

        return df


    def populate_tickers(self, exchanges: list, asset_types: list, timeframes: list):
        """ Downloading all available tickers.
        
        This method downloads all ticker from 'suported_tickers.csv'
        that match exchange and asset type (ETF, Stock, etc.) limitations
        for each bar frequency in timeframes. Uses parallelization.

        Parameters
        ----------
        exchanges : list 
            List of strings that specify an exchange, e.g. ["NASDAQ"]
        asset_types : list
            List of strings that specify types of assets to download,
            e.g. ["ETF"], ["ETF", "Stock"]
        timeframes : list
            List of integers that specify the timeframes to download,
            in minutes, e.g. [5, 60, 1440] 
        
        Returns
        -------
        None
        """
        # Get list of supported tickers
        self._print(msg="Populating TICKERS!")
        df = pl.read_csv(f"{self.path}/TickerInfo/supported_tickers.csv").to_pandas()

        # Filtering for exchanges 
        ind = np.array([df.exchange.to_numpy() == exchange 
                        for exchange in exchanges]).T.sum(axis=1) > 0
        symbols = df.loc[ind, ('ticker', 'assetType', 'startDate', 'endDate')]

        # Filtering for asset types
        if asset_types:
            ind = np.array([symbols.assetType.to_numpy() == asset_type 
                            for asset_type in asset_types]).T.sum(axis=1) > 0
            symbols = symbols.loc[ind]
        
        # Creating folders
        for asset_type in symbols.assetType.unique():
            self._create_folder(f"/Data/{asset_type}")
            for timeframe in timeframes:
                self._create_folder(f"/Data/{asset_type}/{timeframe}")

        # Gathering all information and passing it as workload to Pool
        l = []
        for ticker, asset_type, startdate, enddate in symbols.to_numpy()[0:10]:
            for timeframe in timeframes:
                if str(startdate) != "NaT":
                   l.append((ticker, asset_type, timeframe, startdate, enddate))
        
        with Pool(self.cores) as p:
            p.starmap(self._populate_ticker, l)
        
        return None

    def _populate_ticker(self,
                         ticker: str,
                         asset_type: str,
                         timeframe: str, 
                         startdate: str, 
                         enddate: str):
        """ Helper function that downloads a single ticker"""

        try:
            if not os.path.isfile(f"{self.path}/Data/{asset_type}/{timeframe}/{ticker}_{timeframe}.csv"):
                df = self.download_ticker(ticker=ticker, 
                                          startdate=startdate, 
                                          enddate=enddate, 
                                          timeframe=timeframe)
                if type(df) == type(None):
                    self._print(msg=f"{asset_type}: {ticker}_{timeframe} not downloaded!") 
                    return None 
                else:
                    d1 = pd.to_datetime(df.date.iloc[0]).date()
                    d2 = pd.to_datetime(df.date.iloc[-1]).date()
                    pl.from_pandas(df).to_csv(f"{self.path}/Data/{asset_type}/{timeframe}/{ticker}_{timeframe}.csv")
                    self._print(msg=f"{asset_type}: {ticker}_{timeframe}\tfrom {d1} till {d2} successfully downloaded!")
            else:
                self._print(msg=f"{asset_type}: {ticker}_{timeframe} already exists!")

        except Exception as e:
            self._print(msg=f"{asset_type}: {ticker}_{timeframe} failed! - Reason: {e}")
        
        return None

    def download_crypto(self, ticker: str, startdate: str, enddate: str, timeframe: int):
        """ This function downloads a crypto pair for a given period and sampling frequency.

        Parameters
        ----------
            ticker : str
                Specifies the symbol to download
            startdate :str
                Date that specifies the first date of interest, fmt: 'YYYY-mm-dd HH:MM:ss'
            enddate : str
                Date that specifies the last date of interest, fmt: 'YYYY-mm-dd HH:MM:ss'
            timeframe : str
                Specifies the bar frequency. In minutes
        
        Returns
        -------
        df : pandas.DataFrame
                Table containing dates and price data (ohlcv+)
        """
        startdate = pd.to_datetime(startdate)
        enddate = pd.to_datetime(enddate)
        span = (startdate, enddate)
        intervals = pd.date_range(start=span[0], end=span[-1], freq=f"{timeframe}min")

        l = []
        n = 10000
        d = len(intervals)//n

        if len(intervals) > n:
            for i in range(n, d*n+1, n):
                l.append(self._get_crypto(ticker, intervals[i], intervals[i+1], timeframe))
            l.append(self._get_crypto(ticker=ticker, 
                                      startdate=str(pd.to_datetime(l[-1].date.iloc[-1])), 
                                      enddate=str(enddate), 
                                      timeframe=timeframe))
            df = pd.concat(l) 
        else:
            df = self._get_crypto(ticker=ticker, 
                                  startdate=str(startdate), 
                                  enddate=str(enddate), 
                                  timeframe=timeframe)

        if type(df) == type(None):
            return None
        
        cols = ('date', 'open', 'high', 'low', 'close', 'volume', 'tradesDone', 'volumeNotional')
        df = df.loc[:, cols]
        df.drop_duplicates(subset=['date'], inplace=True)
        df = self._reset_idx(df)

        return df

    def populate_cryptos(self, quote_currencies: list, timeframes: list):
        """ Downloading all available crypto pairs.
        
        This method downloads all crypto pairs from 'supported_cryptos.csv'
        that match the quote currency (usd, eur, etc.) limitations
        for each bar frequency in timeframes. Uses parallelization.

        Parameters
        ----------
        quote_currencies : list
            List of strings that specify the quote currencies to download,
            e.g. ["eur"], ["usd", "rub"]
        timeframes : list
            List of integers that specify the timeframes to download,
            in minutes, e.g. [5, 60, 1440] 
        
        Returns
        -------
        None
        """
        # Getting supported crypto pairs
        self._print(msg="Populating Cryptos!")
        df = pl.read_csv(self.path + f"/TickerInfo/supported_crypto.csv").to_pandas()
        symbols = df.loc[:, ('ticker', 'baseCurrency', 'quoteCurrency')]

        # Filter target quote currencies
        if quote_currencies:
            ind = np.array([df.loc[:, 'quoteCurrency'] == currency 
                            for currency in quote_currencies]).T.sum(axis=1) > 0
            symbols = symbols.loc[ind]

        # Creating folders
        self._create_folder("Data/Crypto")
        for timeframe in timeframes:
            self._create_folder(f"Data/Crypto/{timeframe}")

        # Gathering all information and passing it as workload to Pool
        l = []
        for ticker, baseCurrency, quoteCurrency in symbols.to_numpy()[0:30]:
                l.append((ticker, timeframes, baseCurrency, quoteCurrency))
               
        with Pool(self.cores) as p:
            p.starmap(self._populate_one_crypto, l)
        
        return None

    def _populate_one_crypto(self, 
                             ticker: str, 
                             timeframes: list, 
                             baseCurrency: str, 
                             quoteCurrency: str):
        """ Helper function that downloads a single crypto pair"""

        try:
            # Check if any timeframe for the pair does not exist
            counter = 0
            for timeframe in timeframes:
                if not os.path.isfile(f"{self.path}/Data/Crypto/{timeframe}/{baseCurrency}_{quoteCurrency}_{timeframe}.csv"):
                    counter += 1

            # If data needs to be downloaded, getting start and enddate of the pair by
            # initial download of weekly data
            if counter > 0:
                df = self._get_crypto(ticker=str(ticker),
                                    startdate=pd.to_datetime("2000-01-01"),
                                    enddate=pd.to_datetime(datetime.today()),
                                    timeframe=str(1440*7))
                if type(df) == type(None):
                    self._print(msg=f"Crypto: {ticker}_{timeframe} not downloaded!") 
                    return None 

                startdate = pd.to_datetime(df.date.iloc[0]).tz_localize(None)
                enddate = pd.to_datetime(datetime.now())
            else:
                self._print(msg=f"Crypto: {ticker} all timeframes already exists!")
                return None

            # Download all timeframes of the pair
            for timeframe in timeframes:
                if not os.path.isfile(f"{self.path}/Data/Crypto/{timeframe}/{baseCurrency}_{quoteCurrency}_{timeframe}.csv"):
                    df = self.download_crypto(ticker=ticker,
                                              startdate=startdate, 
                                              enddate=enddate, 
                                              timeframe=timeframe)
                    if type(df) == None:
                        self._print(msg=f"Crypto: {ticker}_{timeframe} not downloaded!") 
                        return None 
                    else:
                        d1 = pd.to_datetime(df.date.iloc[1]).date()
                        d2 = pd.to_datetime(df.date.iloc[-1]).date()
                        pl.from_pandas(df).to_csv(f"{self.path}/Data/Crypto/{timeframe}/{baseCurrency}_{quoteCurrency}_{timeframe}.csv")
                        self._print(msg=f"Crypto: {baseCurrency}_{quoteCurrency}_{timeframe}\t\tfrom {d1} till {d2} successfully downloaded!")
                else:
                    self._print(msg=f"Crypto: {ticker}_{timeframe} already exists!")

        except Exception as e:
            self._print(msg=f"Crypto: {ticker} failed! - Reason: {e}")
            pass           
        
        return None 

    def update_tickers(self, asset_types: list, timeframes: list):
        assert False, "Not Finished!"
        """ This function downloads a crypto pair for a given period and sampling frequency.

        Parameters
        ----------
            ticker : str
                Specifies the symbol to download
            startdate :str
                Date that specifies the first date of interest, fmt: 'YYYY-mm-dd HH:MM:ss'
            enddate : str
                Date that specifies the last date of interest, fmt: 'YYYY-mm-dd HH:MM:ss'
            timeframe : str
                Specifies the bar frequency. In minutes
        
        Returns
        -------
        df : pandas.DataFrame
                Table containing dates and price data (ohlcv+)
        """
        # Geting supported tickers
        self._print(msg=f"Updating TICKERS!")
        df = pl.read_csv(f"{self.path}/TickerInfo/supported_tickers.csv").to_pandas()
        symbols = df.loc[:, ('ticker', 'assetType', 'startDate', 'endDate')]
        symbols = symbols.set_index(keys=['ticker'])
        
        for asset_type in asset_types:
            ind = symbols.assetType.to_numpy() == asset_type
            symbols = symbols.loc[ind]

            for timeframe in timeframes:
                folder = f"{self.path}/Data/{asset_type}/{timeframe}"
                for _, _, files in os.walk(folder):
                    for file in files:
                        try:
                            ticker = file.split("_")[0]
                            df = pl.read_csv(f"{folder}/{file}").to_pandas()
                            startdate = pd.to_datetime(df.date.iloc[-1])
                            enddate = pd.to_datetime(symbols.loc[ticker, 'endDate'])

                            if startdate >= enddate:
                                self._print(msg=f"{type}: {file.split('.')[0]} at {startdate} up to date!")
                            else:
                                temp_df = self.download_ticker(ticker=ticker, 
                                                                timeframe=timeframe, 
                                                                startdate=startdate, 
                                                                enddate=pd.to_datetime(datetime.now()))
                                df = pd.concat([df, temp_df], axis=0)
                                df.date = pd.to_datetime(df.date).dt.tz_localize(None)
                                df.drop_duplicates(subset=['date'], inplace=True)
                                df = df.reset_index().drop(columns="index")                          
                                pl.from_pandas(df).to_csv(f"{folder}/{file}")
                                self._print(msg=f"{type}: {file.split('.')[0]} from {startdate} to {df.date.iloc[-1]}!")

                        except Exception as e:
                            self._print(msg=f"{type}: {file.split('.')[0]} failed! - Reason: {e}")

    def update_cryptos(self, timeframes: list):
        assert False, "Not Finished!"
        """ This function downloads a crypto pair for a given period and sampling frequency.

        Parameters
        ----------
            ticker : str
                Specifies the symbol to download
            startdate :str
                Date that specifies the first date of interest, fmt: 'YYYY-mm-dd HH:MM:ss'
            enddate : str
                Date that specifies the last date of interest, fmt: 'YYYY-mm-dd HH:MM:ss'
            timeframe : str
                Specifies the bar frequency. In minutes
        
        Returns
        -------
        df : pandas.DataFrame
                Table containing dates and price data (ohlcv+)
        """
        self._print(msg=f"Updating CRYPTOS!")
        for timeframe in timeframes:
            folder = f"{self.path}/Data/Crypto/{timeframe}"
            for _, _, files in os.walk(folder):
                for file in files:
                    try:
                        ticker = file.split("_")[0]
                        df = pl.read_csv(f"{folder}/{file}").to_pandas()
                        startdate = pd.to_datetime(df.date.iloc[-1]).tz_localize(None)
                        enddate = pd.to_datetime(datetime.now())

                        if startdate >= enddate:
                            self._print(msg=f"Crypto: {file.split('.')[0]} at {startdate} up to date!")
                        else:
                            temp_df = self.download_crypto(ticker=ticker, 
                                                            timeframe=timeframe, 
                                                            startdate=startdate, 
                                                            enddate=enddate)
                            df = pd.concat([df, temp_df], axis=0)
                            df.drop_duplicates(subset=['date'], inplace=True)
                            df = df.reset_index().drop(columns="index")
                            pl.from_pandas(df).to_csv(f"{folder}/{file}")
                            self._print(msg=f"Crypto: {file.split('.')[0]} from {startdate} to {df.date.iloc[-1]}!")
                    except Exception as e:
                        self._print(msg=f"Crypto: {ticker}_{timeframe} failed! - Reason: {e}")
                        pass  

    def smooth_data(self):
        """ Standardizing column order, removing duplicates """

        # Getting available files list or calling function
        self._print(msg="Smoothing Data!")
        if not os.path.isfile(f"{self.path}/Temp/available_files.csv"):
            self.get_available_files()
        ticker_list = pl.read_csv(f"{self.path}/Temp/available_files.csv").to_pandas()

        # creating tuple with infos to all av. files and passing it to Pool
        l =  [(typ, symbol, timeframe, path) 
                    for typ, symbol, timeframe, path in ticker_list.to_numpy()]
        with Pool(12) as p:
            p.starmap(self._smooth_one, l)

    def _smooth_one(self, typ: str, symbol: str, timeframe: int, path: str):
    
        col_order = ("date","open","high","low","close","volume")
        if typ == "Crypto" and timeframe > 1:
            col_order = (
                "date","open","high","low","close","volume","tradesDone","volumeNotional"
                )
        else:    
            if timeframe >= 1440:
                col_order = (
                    "date","open","high","low","close","volume",\
                    "adjOpen","adjHigh","adjLow","adjClose","adjVolume",\
                    "divCash","splitFactor"
                    )
        df = pl.read_csv(path).to_pandas()
        df = df.loc[:,col_order].drop_duplicates(subset=['date'])
        pl.from_pandas(df).to_csv(path) 
        self._print(msg=f"{typ}:\t{symbol}_{timeframe} smoothed!")

    def _get_response(self, url: str):
        """ Accessing API at endpoint specified by url """ 

        # print(self.host + url + "&token=" + self.api_key)
        # input("s")
        while True:
            requestResponse = get(f"{self.host}{url}&token={self.api_key}",
                                        headers=self.headers)
            requestResponse = requestResponse.json() 
            if type(requestResponse) == dict:
                if "Error" in requestResponse["detail"]:
                    self._print(msg=f"Data limit reached! Waiting 60 minutes.")
                    sleep(60)
            else: 
                break
        return requestResponse                  
    
    def _get_crypto(self, ticker: str, startdate: str, enddate: str, timeframe: int):

        url = f"tiingo/crypto/prices?tickers={ticker}&startDate={startdate}\
                &endDate={enddate}&resampleFreq={timeframe}min"
        response = self._get_response(url)
        if len(response) > 0:
            return pd.DataFrame.from_dict(dict(response[0])['priceData'])
        else:
            return None

    def _get_eod(self, ticker: str, startdate: str, enddate: str = None):
        if enddate:
            url = f"tiingo/daily/{ticker}/prices?startDate={startdate}&endDate={enddate}"
        else:
            url = f"tiingo/daily/{ticker}/prices?startDate={startdate}"
        response = self._get_response(url)
        
        return pd.DataFrame.from_dict(response)

    def _get_iex(self, ticker: str, startdate: str, enddate: str, timeframe: int):

        url = f"iex/{ticker}/prices?startDate={startdate}&endDate={enddate}\
                &columns=open,high,low,close,volume&resampleFreq={timeframe}min"
        response = self._get_response(url)
        
        return pd.DataFrame.from_dict(response)

    
if __name__ == "__main__":

    folder = "C:/Users/Henry/OneDrive - rub.de/Desktop/Neuer Ordner"
    api_key = "c582fe1982f846e9da78961fb06d8063ef4a55b0"
    exchanges = ["NYSE", "NASDAQ", "NYSE ARCA", "BATS"]
    asset_types = ["ETF", "Stock"]
    timeframes = [1440]
    td = TiingoDownloader(api_key=api_key, folder=folder)
    td.populate_tickers(exchanges=exchanges, asset_types=asset_types, timeframes=timeframes)
    # td.populate_cryptos(quote_currencies=["usd", "eur"], timeframes=timeframes)
    # td.update_tickers(assetTypes=assetTypes, timeframes=timeframes)
    # td.update_cryptos(timeframes=timeframes)
    # td.smooth_data()
    # a = td.download_crypto("5aaveusd", "2021-01-01", "2021-02-01", 1440)
    # print(a)
    
  