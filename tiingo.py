import os
import requests
import numpy as np
import pandas as pd
import polars as pl
from pandas.tseries.offsets import BDay
from datetime import datetime
from multiprocessing import Pool, cpu_count
from lib.baseClass import BaseClass

class TiingoDownloader(BaseClass):
    """ This class provides a solution to automatically 
        downloading data from Tiingos RESTful API. """

    def __init__(self, api_key: str, folder=None, verbose=True):
        """ Class Initialization"""

        super().__init__(folder=folder, verbose=verbose)
        self.verbose = verbose
        self.api_key = api_key
        self.cores = cpu_count() 
        self.host = "https://api.tiingo.com/"
        self.headers = {'Content-Type': 'application/json'}
        self.check_ticker()
    
    def _get_response(self, url: str):
        """ Accessing API at endpoint specified by url """ 
        # print(self.host + url + "&token=" + self.api_key)
        # input("s")
        if "?" in url:
            requestResponse = requests.get(f"{self.host}{url}&token={self.api_key}",
                                            headers=self.headers)
        else:
            requestResponse = requests.get(f"{self.host}{url}?token={self.api_key}",
                                            headers=self.headers)
        requestResponse = requestResponse.json() 

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

    def check_ticker(self, force_update=False):
        """ Gets up to date ticker information from API and
            saves results to 'self.path/TickerInfo/' """

        # Check last update to reduce API data load
        last_update = self._last_update(force_update=force_update)
        if last_update and last_update == pd.to_datetime(datetime.now().date()):
            self._print(msg=f"Ticker info up to date with last update: {last_update.date()}")
            sup_tickers = pd.read_csv(f"{self.path}/TickerInfo/supported_tickers.csv")
            sup_cryptos = pd.read_csv(f"{self.path}/TickerInfo/supported_crypto.csv")

            return sup_tickers, sup_cryptos
                   
        # creating necessary folders in specified path
        folders = ["Temp", "TickerInfo", "PriceData"]
        for folder in folders: 
            self._create_folder(folder)

        # Getting list of all supported STOCK, ETF, and FUND ticker 
        self._print(msg=f"Getting available tickers...")
        sup_tickers = pd.read_csv(
            "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
            )
        pl.from_pandas(sup_tickers).to_csv(f"{self.path}/TickerInfo/supported_tickers.csv")
        self._print(msg=f"Available tickers: {len(sup_tickers)}")

        # Getting list of all supported CRYPTO pairs  
        self._print(msg=f"Getting available cryptos...")
        response = self._get_response("/tiingo/crypto?")
        sup_cryptos = pd.DataFrame.from_dict(response).drop('description', axis=1)
        sup_cryptos = self._reset_idx(sup_cryptos)
        sup_cryptos = sup_cryptos.loc[:, ('ticker', 'name', 'baseCurrency', 'quoteCurrency')]
        sup_cryptos = sup_cryptos.sort_values('ticker')
        pl.from_pandas(sup_cryptos).to_csv(f"{self.path}/TickerInfo/supported_crypto.csv")
        self._print(msg=f"Available cryptos: {len(sup_cryptos)}")   

        with open(f"{self.path}/Temp/last_update.txt", "w") as f:
            f.write(str(datetime.now().date()))
            f.close()

        return sup_tickers, sup_cryptos

    def _last_update(self, force_update):
        # Check last update to reduce API data load
        if not force_update:
            if os.path.isfile(f"{self.path}/Temp/last_update.txt"):
                today = pd.to_datetime(datetime.now().date())

                with open(f"{self.path}/Temp/last_update.txt", "r") as f:
                    last_update = pd.to_datetime(f.read())

                return last_update
        
        return None

    def get_available_files(self):
        """ Screening for available files
        
        This method walks through every subdirectory found in 'self.path', gathers 
        file information and returns it for all available files. Saves results to
        'self.path/Temp/'.

        Parameters
        ----------
        
        Returns
        -------
        hits: pandas.DataFrame
            Ticker information for every available files.
        """

        self._print(msg=f"Getting available files!")
        folder = f"{self.path}/PriceData/"
        d = {}

        for root, _, files in os.walk(folder):
            d[root] = files

        l = []
        for key, files in d.items():
            if len(files) > 0 and files:
                for file in files:
                    l.append((key, file))
        
        with Pool(self.cores) as p:
            hits = p.starmap(self._get_one_price_data, l)

        if len(hits) > 0:
            cols = ["symbol", "type", "timeframe", "startdate", "enddate",
                    "missing", "path", "fundamentals_path"]
            hits = pd.DataFrame(hits, columns=cols, index=None)
            hits = hits.reset_index().drop(columns="index")
            pl.from_pandas(hits).to_csv(f"{self.path}/Temp/available_files.csv")
            self._print(msg=f"{len(hits)} files found!")
        else:
            self._print(msg=f"No available files found!")
        
        return hits

    def _get_one_price_data(self, key, file):
        """ Helper function to access one price data file."""

        path = f"{key}/{file}"
        df = pl.read_csv(path).to_pandas()
        symbol = str(file.split("_")[0])
        typ, t = key.split("/")[-1].split("\\")
        start, end = df.date.iloc[[0, -1]]
        missing = df.isnull().sum().sum()

        fundamentals_path = None
        if os.path.isfile(f"{self.path}/Fundamentals/{typ}/{symbol}.txt"):
            fundamentals_path = f"{self.path}/Fundamentals/{typ}/{symbol}.txt"

        return [symbol, typ, t, start, end, missing, path, fundamentals_path]
    
    def download_ticker(self, ticker: str, timeframe: int, startdate: str, enddate: str): 
        """ This function downloads a ticker for a given period and sampling frequency.

        Parameters
        ----------
            ticker : str
                Specifies the symbol to download
            timeframe : str
                Specifies the bar frequency. In minutes
            start :str
                Date that specifies the first date of interest, fmt: 'YYYY-mm-dd HH:MM:ss'
            end : str
                Date that specifies the last date of interest, fmt: 'YYYY-mm-dd HH:MM:ss'
        
        Returns
        -------
        df : pandas.DataFrame
                Table containing dates and price data (ohlcv+)
        """
        # Switching between API endpoints
        if timeframe == 1440:
            df = self._get_eod(ticker=ticker, startdate=startdate, enddate=enddate)
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
    
                if type(l[-1]) == type(None) or l[-1].empty:
                    return None
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

        if type(df) == type(None) or df.empty:
            return None

        df.drop_duplicates(subset=['date'], inplace=True)
        df = df.reset_index().drop(columns="index")

        return df

    def populate_tickers(self, exchanges: list, asset_types: list, timeframes: list):
        """ Downloading all available tickers.
        
        This method downloads all ticker from 'suported_tickers.csv'
        that match exchange (NASDAQ, etc.) and asset type (ETF, Stock, etc.) 
        limitations for each bar frequency in timeframes. Uses parallelization.

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

        # Filter ticker that already have been tried to download and failed
        try:
            na_files = pd.read_csv(f"{self.path}/Temp/not_available_files.csv", header=None)
        except:
            na_files = pd.DataFrame([[None, None, None]])

        # Filtering for asset types
        if asset_types:
            ind = np.array([symbols.assetType.to_numpy() == asset_type 
                            for asset_type in asset_types]).T.sum(axis=1) > 0
            symbols = symbols.loc[ind]
        
        # Creating folders
        for asset_type in symbols.assetType.unique():
            self._create_folder(f"/PriceData/{asset_type}")
            for timeframe in timeframes:
                self._create_folder(f"/PriceData/{asset_type}/{timeframe}")

        # Gathering all information and passing it as workload to Pool
        l = []
        for ticker, asset_type, startdate, enddate in symbols.to_numpy():
            for timeframe in timeframes:
                if str(startdate) != "NaT":
                    flag = self._already_tried(na_files, ticker, timeframe)
                    if flag: 
                        continue
                    else:
                        l.append((ticker, asset_type, timeframe, startdate, enddate))
        
        with Pool(self.cores) as p:
            p.starmap(self._populate_ticker, l)
        
        self.get_available_files()

        return None

    def _populate_ticker(self, ticker: str, asset_type: str, timeframe: str, start: str, end: str):
        """ Helper function that downloads a single ticker"""

        try:
            if not os.path.isfile((
                    f"{self.path}/PriceData/{asset_type}/"
                    f"{timeframe}/{ticker}_{timeframe}.csv")):
                df = self.download_ticker(ticker=ticker, 
                                            startdate=start, 
                                            enddate=end, 
                                            timeframe=timeframe)

                if type(df) == type(None) or df.empty:
                    self._print_ticker_info("NOT available!", asset_type, ticker, timeframe)
                    self._write_tried(asset_type, ticker, timeframe)
                    return None
                
                d1 = pd.to_datetime(df.date.iloc[0]).date()
                d2 = pd.to_datetime(df.date.iloc[-1]).date()
                pl.from_pandas(df).to_csv((
                    f"{self.path}/PriceData/{asset_type}/"
                    f"{timeframe}/{ticker}_{timeframe}.csv"
                    ))
                self._print_ticker_info(f"from {d1} till {d2} successfully downloaded!", 
                                        asset_type, ticker, timeframe)
            else:
                self._print_ticker_info("already exists!", asset_type, ticker, timeframe)

        except Exception as e:
            self._print(msg=f"{asset_type}:\t{ticker}_{timeframe} failed! - Reason: {e}")
        
        return None

    def download_crypto(self, ticker: str, startdate: str, enddate: str, timeframe: int):
        """ This function downloads a crypto pair for a given period and sampling frequency.

        Parameters
        ----------
            ticker : str
                Specifies the symbol to download
            start :str
                Date that specifies the first date of interest, fmt: 'YYYY-mm-dd HH:MM:ss'
            end : str
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
        n = 5000               # Max number of rows of Tiingo API requests
        d = len(intervals)//n   # Number of iterations for the loop

        # Batched download if daterange is longer than n
        if len(intervals) > n:
            for i in range(0, d*n, n):
                l.append(self._get_crypto(ticker, intervals[i], intervals[i+n], timeframe))
            if type(l[-1]) == type(None) or l[-1].empty:
                return None

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

        if type(df) == type(None) or df.empty:
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
        self._create_folder("PriceData/Crypto")
        for timeframe in timeframes:
            self._create_folder(f"PriceData/Crypto/{timeframe}")

        # Filter ticker that already have been tried to download and failed
        try:
            na_files = pd.read_csv(f"{self.path}/Temp/not_available_files.csv", header=None)
        except:
            na_files = pd.DataFrame([[None, None, None]])
            
        # Gathering all information and passing it as workload to Pool
        l = []
        for ticker, baseCurrency, quoteCurrency in symbols.to_numpy():
            l.append((ticker, timeframes, baseCurrency, quoteCurrency, na_files.copy()))
               
        with Pool(self.cores) as p:
            p.starmap(self._populate_one_crypto, l)
        
        self.get_available_files()

        return None

    def _populate_one_crypto(self, ticker: str, timeframes: list, baseCurrency: str,
                            quoteCurrency: str, na_files: pd.DataFrame):
        """ Helper function that downloads a single crypto pair"""

        try:
            # Check if any timeframe for the pair does not exist
            counter = 0
            for timeframe in timeframes:
                if not os.path.isfile((
                        f"{self.path}/PriceData/Crypto/{timeframe}/"
                        f"{baseCurrency}_{quoteCurrency}_{timeframe}.csv")):
                    counter += 1

            # If data needs to be downloaded, getting start and enddate of the pair by
            # initial download of weekly data
            if counter > 0:
                df = self._get_crypto(ticker=str(ticker),
                                    startdate=pd.to_datetime("2000-01-01"),
                                    enddate=pd.to_datetime(datetime.today()),
                                    timeframe=str(1440*7))
                if type(df) == type(None):
                    flag = self._already_tried(na_files, ticker, 1440*7, verbose=False)
                    if not flag:  
                        self._write_tried("Crypto", ticker, 1440*7)
                    else:
                        for timeframe in timeframes:
                            self._print_ticker_info("NOT available!", "Crypto", ticker, timeframe)
                    return None

                startdate = pd.to_datetime(df.date.iloc[0]).tz_localize(None)
                enddate = pd.to_datetime(datetime.now())
            else:
                for timeframe in timeframes:
                    self._print_ticker_info("already exists!", "Crypto", 
                                            ticker, timeframe)
                return None

            # Download all timeframes of the pair
            for timeframe in timeframes:

                flag = self._already_tried(na_files, ticker, timeframe)
                if flag: continue

                if not os.path.isfile((
                    f"{self.path}/PriceData/Crypto/{timeframe}/"
                    f"{baseCurrency}_{quoteCurrency}_{timeframe}.csv")):
                    df = self.download_crypto(ticker=ticker,
                                            startdate=startdate, 
                                            enddate=enddate, 
                                            timeframe=timeframe)

                    if type(df) == type(None) or df.empty:
                        self._print_ticker_info("NOT available!", "Crypto", ticker, timeframe) 
                        self._write_tried("Crypto", ticker, timeframe)
                        return None

                    else:
                        d1 = pd.to_datetime(df.date.iloc[1]).date()
                        d2 = pd.to_datetime(df.date.iloc[-1]).date()
                        pl.from_pandas(df).to_csv((
                            f"{self.path}/PriceData/Crypto/{timeframe}/"
                            f"{baseCurrency}_{quoteCurrency}_{timeframe}.csv"
                        ))
                        self._print_ticker_info(f"from {d1} till {d2} successfully downloaded!", 
                                                "Crypto", ticker, timeframe)
                else:
                    self._print_ticker_info("already exists!", "Crypto", ticker, timeframe)

        except Exception as e:
            self._print_ticker_info(f"failed! - Reason: {e}", "Crypto", ticker, timeframe)           
        
        return None 

    def update_tickers(self, asset_types: list, timeframes: list):
        """ This method updates all ticker files of a given asset type and bar frequency.

        Parameters
        ----------
            asset_types : list
                Specifies the type of tickers that should be updated (e.g., ETF, Stock)
            timeframes :list
                Specifies the bar frequency. In minutes
        
        Returns
        -------
        None
        """
        # Geting supported tickers
        self._print(msg=f"Updating TICKERS!")
        av_files = pl.read_csv(f"{self.path}/Temp/available_files.csv").to_pandas()        
        sup_tickers = pl.read_csv(f"{self.path}/TickerInfo/supported_tickers.csv").to_pandas()

        # Filtering for asset type and timeframes
        ind1 = np.array([av_files.type.to_numpy() == asset_type 
                        for asset_type in asset_types]).T.sum(axis=1) > 0
        ind2 = np.array([av_files.timeframe.to_numpy() == timeframe 
                        for timeframe in timeframes]).T.sum(axis=1) > 0
        ind  = np.logical_and(ind1, ind2)
        av_files = av_files.loc[ind, ["type", "symbol", "timeframe", "startdate", "path"]]

        # Filtering for asset type and timeframes
        sup_tickers = sup_tickers.loc[:, ["ticker", "endDate"]]
        ind = sup_tickers.set_index("ticker").index.intersection(av_files.symbol.unique())
        sup_tickers = sup_tickers.set_index("ticker").loc[ind, :]

        l = []
        for asset_type, symbol, timeframe, startdate, path in av_files.to_numpy():
            if symbol not in sup_tickers.index:
                self._print_ticker_info(f"no longer supported! Deleted!", 
                                        asset_type, symbol, timeframe)
                os.remove(path)
                continue
            if type(sup_tickers.loc[symbol, "endDate"]) == pd.Series:
                enddate = pd.to_datetime(sup_tickers.loc[symbol, "endDate"]).max()
            else:
                enddate = sup_tickers.loc[symbol, "endDate"]
            l.append((asset_type, symbol, timeframe, startdate, enddate, path))
        
        with Pool(self.cores) as p:
            p.starmap(self._update_one_ticker, l)

        return None

    def _update_one_ticker(self, asset_type, symbol, timeframe, startdate, enddate, path):
        """ Helper function to update a single ticker"""
        try:
            df = pl.read_csv(path).to_pandas()
            startdate = pd.to_datetime(df.date.iloc[-1])
            enddate = pd.to_datetime(enddate)

            if enddate < pd.to_datetime(datetime.now().date()) - BDay(90):
                self._print_ticker_info(f"ticker not active anymore!",
                                        asset_type, symbol, timeframe)
                return None

            if timeframe >= 1440:
                dates = pd.DataFrame(None, index=pd.DatetimeIndex(df.date))
                dates["year"] = dates.index.year
                dates["month"] = dates.index.month
                previous_date = dates.iloc[-1, :]

                for i in reversed(range(len(dates))):
                    if dates.iloc[i, 1] < previous_date.month - 1:
                        startdate = dates.iloc[i + 1].name
                        break
                            
                update = self.download_ticker(ticker=symbol, 
                                            timeframe=timeframe, 
                                            startdate=startdate, 
                                            enddate=pd.to_datetime(datetime.now()))
                df = self._merge_update(df, update)
                pl.from_pandas(df).to_csv(path)
                self._print_ticker_info(f"updated from {startdate} to {df.date.iloc[-1]}!", 
                                        asset_type, symbol, timeframe)                        
            else:
                update = self.download_ticker(ticker=symbol, 
                                            timeframe=timeframe, 
                                            startdate=startdate, 
                                            enddate=pd.to_datetime(datetime.now()))
                df = self._merge_update(df, update)
                pl.from_pandas(df).to_csv(path)
                self._print_ticker_info(f"updated from {startdate} to {df.date.iloc[-1]}!", 
                                        asset_type, symbol, timeframe) 

            return None

        except Exception as e:
            self._print_ticker_info(f"failed due to: {e}", asset_type, symbol, timeframe)
            return None

    def _merge_update(self, df, update):

        df = pd.concat([df, update], axis=0)     
        df.date = pd.to_datetime(df.date).dt.tz_localize(None)
        df.drop_duplicates(subset=['date'], keep="last", inplace=True)
        df = df.reset_index().drop(columns="index") 

        return df

    def update_cryptos(self, timeframes: list):
        """ This method updates every crypto pair price data for bar frequencies
            of interest.

        This method updates crypto pair price data for every available
        price data that matches the bar frequencies of interest. Results
        will be stored in ".../PriceData/Crypto/..." as .csv file.
        Uses parallelization.

        Parameters
        ----------
            timeframes : list
                List of integers that specify the bar frequencies of interest.
        
        Returns
        -------
        None
        """
        # Geting supported tickers
        self._print(msg=f"Updating CRYPTO!")
        av_files = pl.read_csv(f"{self.path}/Temp/available_files.csv").to_pandas()  

        # Filtering for asset type and timeframes
        ind1 = np.array(av_files.type.to_numpy() == "Crypto")       
        ind2 = np.array([av_files.timeframe.to_numpy() == timeframe 
                        for timeframe in timeframes]).T.sum(axis=1) > 0
        ind  = np.logical_and(ind1, ind2)     
        av_files = av_files.loc[ind, ["type", "symbol", "timeframe", "enddate", "path"]]

        l = []
        for asset_type, symbol, timeframe, enddate, path in av_files.to_numpy():
            l.append((asset_type, symbol, timeframe, enddate, path))

        with Pool(self.cores) as p:
            p.starmap(self._update_one_crypto, l)
        
        self.get_available_files()
        
        return None
    
    def _update_one_crypto(self, asset_type, symbol, timeframe, enddate, path):
        """ Helper function to update a single crypto pair"""
        try:
            df = pl.read_csv(path).to_pandas()
            startdate = pd.to_datetime(df.date.iloc[-1]).tz_localize(None)
            enddate = pd.to_datetime(datetime.now())

            if startdate >= enddate:
                self._print_ticker_info(f"{startdate} up to date!",
                                        asset_type, symbol, timeframe)
            else:
                update = self.download_crypto(ticker=symbol, 
                                              timeframe=timeframe, 
                                              startdate=startdate, 
                                              enddate=enddate)
                df = self._merge_update(df, update)
                pl.from_pandas(df).to_csv(path)
                self._print_ticker_info(f"updated from {startdate} to {df.date.iloc[-1]}!", 
                                        asset_type, symbol, timeframe) 

        except Exception as e:
            self._print_ticker_info(f"failed due to: {e}", asset_type, symbol, timeframe)
        
        return None

    def smooth_data(self):
        """ Standardizing column order, removing duplicates """

        # Getting available files list or calling function
        self._print(msg="Smoothing Data!")
        if not os.path.isfile(f"{self.path}/Temp/available_files.csv"):
            self.get_available_files()
        ticker_list = pl.read_csv(f"{self.path}/Temp/available_files.csv").to_pandas()

        # creating tuple with infos to all av. files and passing it to Pool
        l =  [(typ, symbol, timeframe, path) 
                    for typ, symbol, timeframe, _, _, _, path in ticker_list.to_numpy()]
        with Pool(self.cores) as p:
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
        del df

    def populate_fundamentals(self):

        definitions = pl.read_csv(
            f"{self.path}/Fundamentals/fundamentals_definitions.csv"
            ).to_pandas()

        symbol = "msft"
        response = self._get_response(url=f"tiingo/fundamentals/{symbol}/statements")
        df = pd.DataFrame(response)

        fin = []
        for statement_type in definitions.statementType.unique():
            res = []
            for i in range(len(df)):
                if statement_type in df.statementData.iloc[i].keys() and df.iloc[i].quarter != 0:
                    temp_df = pd.DataFrame(df.statementData.iloc[i][statement_type])
                    temp_df["date"] = df.iloc[i].date
                    temp_df = pd.DataFrame(temp_df).pivot(index="date", 
                                                          columns="dataCode", 
                                                          values="value")
                    res.append(temp_df)
            fin.append(pd.concat(res))  
        fin = pd.concat(fin, axis=1)
        fin.to_csv("D:/bla.csv")

if __name__ == "__main__":

    folder = "D:/Tiingo/"
    api_key = "c582fe1982f846e9da78961fb06d8063ef4a55b0"
    exchanges = ["NYSE", "NASDAQ", "NYSE ARCA", "BATS"]
    asset_types = ["ETF", "Stock"]
    timeframes = [1440]
    td = TiingoDownloader(api_key=api_key, folder=folder, verbose=True)
    # td.populate_tickers(exchanges=exchanges, asset_types=asset_types, timeframes=timeframes)
    # td.populate_cryptos(quote_currencies=["usd", "eur"], timeframes=timeframes)
    # td.update_tickers(asset_types=asset_types, timeframes=timeframes)
    # td.update_cryptos(timeframes=timeframes)
    # td.populate_fundamentals()
    for tic in ["adaeur", "btceur", "siaeur"]:
        for i in [1,5,15,60,240]:
            df = td.download_crypto(ticker=tic, startdate="2018-12-01", enddate="2021-10-03", timeframe=i)
            df.to_csv(f"{tic}_{i}.csv", index=False)
    # td.get_available_files()
    # td.smooth_data()

    
  