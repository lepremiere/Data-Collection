import os 
import json
import requests 
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
from lib.baseClass import BaseClass

class EODDownloader(BaseClass):

    def __init__(self, api_key, exchanges, folder=None, verbose=True):
        """ This class provides a solution to automatically 
        downloading data from EOD RESTful API. """

        super().__init__(folder=folder, verbose=verbose)
        self.host = "https://eodhistoricaldata.com/api/"
        self.api_key = api_key
        self.exchanges = exchanges
        self.cores = cpu_count() 
        self.check_ticker()
    
    def _get_response(self, url):
        """ Access point of the EOD-API.

        Parameters
        ----------
        url: string
            Specifies the endpoint to access.

        Returns
        -------
        requestResponse: json
            API response in json format.
        """
        request = f"{self.host}{url}api_token={self.api_key}&fmt=json"
        requestResponse = requests.get(request)

        return requestResponse.json()   

    def check_ticker(self, force_update=False):
        """ Checks/updates available exchanges and its tickers.

        Checks the last update of exchanges and ticker lists and
        updates them if older than the current day. Results will 
        be stored in folder ".../TickerInfo/". Uses parallelization.

        Parameters
        ----------
        force_update : boolean
            If True, exchanges and ticker lists will be updated
            regardless of the last update.
        
        Returns
        -------
        None
        """
        # Check last update to reduce API data load
        last_update = self._last_update(force_update=force_update)
        if last_update and last_update == pd.to_datetime(datetime.now().date()):
            self._print(msg=f"Ticker info up to date with last update: {last_update.date()}")
            return None

        self._print(msg=f"Checking Ticker Lists!")
        folders = ["Temp", "TickerInfo", "TickerInfo/TickerLists"]
        for folder in folders:
            self._create_folder(folder)

        self._print(msg=f"Getting available exchanges!")
        url = "exchanges-list/?"
        exchanges_list = pd.DataFrame.from_dict(self._get_response(url=url))
        pl.from_pandas(exchanges_list).to_csv(f"{self.path}/TickerInfo/exchangesList.csv")
        
        l = []
        exchanges_list = pl.read_csv(f"{self.path}/TickerInfo/exchangesList.csv").to_pandas()
        for name, exchange in exchanges_list.loc[:, ["Name", "Code"]].to_numpy():
            if exchange == "MONEY" or exchange == "BOND":
                continue
            l.append((name, exchange, url))
        
        with Pool(self.cores) as p:
            p.starmap(self._get_ticker_list, l)

        with open(f"{self.path}/Temp/last_update.txt", "w") as f:
            f.write(str(datetime.now().date()))
            f.close()

        return None

    def _get_ticker_list(self, name, exchange, url):
        """ Helper function to access a single exchange 
            and its ticker list."""

        self._print(msg=f"Getting available ticker for {name} - {exchange}...")
        url = f"exchange-symbol-list/{exchange}?"
        ticker_list = pd.DataFrame.from_dict(self._get_response(url=url))
        ticker_list.to_csv(f"{self.path}/TickerInfo/TickerLists/{exchange}.csv")

        return None

    def _last_update(self, force_update):
        """ Helper function that returns the last ticker list update."""

        if not force_update:
            if os.path.isfile(f"{self.path}/Temp/last_update.txt"):
                
                with open(f"{self.path}/Temp/last_update.txt", "r") as f:
                    last_update = pd.to_datetime(f.read())
                    f.close()

                return last_update

        return None

    def get_available_files(self):
        """ Check for available files in the data base.

        This method walks through all price data files,
        gets infos for each symbol and checks if fundamental
        data is available. Results are stored in 
        ".../Temp/available_files.csv". Uses parallelization.

        Parameters
        ----------
        
        Returns
        -------
        None
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
            cols = ["symbol", "type", "timeframe", "start", "end", "path", "fundamentals_path"]
            hits = pd.DataFrame(hits, columns=cols, index=None)
            hits = hits.reset_index().drop(columns="index")
            pl.from_pandas(hits).to_csv(f"{self.path}/Temp/available_files.csv")
            self._get_infos()
            self._print(msg=f"{len(hits)} files found!")
        else:
            self._print(msg=f"No available files found!")
        
        return None
    
    def _get_one_price_data(self, key, file):
        """ Helper function to access one price data file."""

        path = f"{key}/{file}"
        df = pl.read_csv(path).to_pandas()
        symbol = str(file.split("_")[0])
        typ, t = key.split("/")[-1].split("\\")
        start, end = df.date.iloc[[0, -1]]
        start = start.strftime("%Y-%m-%d")
        end = end.strftime("%Y-%m-%d")

        fundamentals_path = None
        if os.path.isfile(f"{self.path}/Fundamentals/{typ}/{symbol}.txt"):
            fundamentals_path = f"{self.path}/Fundamentals/{typ}/{symbol}.txt"

        return [symbol, typ, t, start, end, path, fundamentals_path]

    def _get_infos(self):
        """ Get additional information for each symbol.

        This method gets "name", "isin", "exchange", "path" and
        "fundamental_path" for every available symbol. Uses parallelization.

        Parameters
        ----------
        
        Returns
        -------
        None
        """
        self._print(msg=f"Getting according infos!")
        if not os.path.isfile(f"{self.path}/Temp/available_files.csv"):
            self.get_available_files()
        df = pl.read_csv(f"{self.path}/Temp/available_files.csv").to_pandas()
        # df.set_index("symbol", inplace=True)

        infos = pd.concat([pl.read_csv(f"{self.path}/TickerInfo/TickerLists/{exchange}.csv")\
                            .to_pandas() for exchange in self.exchanges])                
        infos.set_index('Code', inplace=True)
        duplicates = pl.read_csv(f"{self.path}/duplicates.csv").to_pandas()

        l = []
        for typ, symbol, path in df.loc[:, ["type", "symbol", "path"]].to_numpy():   
            l.append((symbol, typ, path, duplicates, infos))

        with Pool(self.cores) as p:
            res = p.starmap(self._get_info, l)
        
        cols = ["symbol", "name", "isin", "exchange"]
        res = pd.DataFrame(res, columns=cols)
        res.set_index("symbol", inplace=True)

        df.set_index("symbol", inplace=True)
        df = df.join(res).reset_index()
        df = df.loc[:, ["symbol", "exchange", "type", "timeframe", "name",
                        "isin", "start", "end", "path", "fundamentals_path"]]
        pl.from_pandas(df).to_csv(f"{self.path}/Temp/available_files.csv")
    
        return None

    def _get_info(self, symbol, typ, path, duplicates, infos):
        """ Helper function to access the additional information 
            of a single symbol"""
        try:
            if symbol in duplicates.Code.values:
                ind = np.logical_and(duplicates.Code == symbol, duplicates.Type == typ)
                name = duplicates.Name.loc[ind].values[0]
                isin = duplicates.Isin.loc[ind].values[0]
                exchange = duplicates.Exchange.loc[ind].values[0]
            else:
                ind = np.logical_and(infos.index == symbol, infos.Type == typ)
                temp_df = infos.loc[ind,:]

                if len(np.shape(temp_df)) == 2:
                    name = temp_df.Name[np.argmax([len(s) for s in temp_df.Name])]
                    isin = temp_df.Isin[np.argmax([len(s) for s in temp_df.Isin])]
                    exchange = temp_df.Exchange[0]
                else:
                    name = temp_df.Name
                    isin = temp_df.Isin
                    exchange = temp_df.Exchange

        except Exception as e:
            self._print_ticker_info(f"no longer supported! Deleted!", typ, symbol, "")
            os.remove(path)
            return []

        return [symbol, name, isin, exchange]
    
    def download_ticker(self, ticker, exchange, timeframe, startdate=None, enddate=None):
        """ Downloads price data for a single ticker.

        This method downloads a ticker from an exchange
        in a specified bar frequency for a given time period.

        Parameters
        ----------
        ticker : str
            The ticker symbol of interest.
        exchange: str
            Exchange to download the ticker from.
        timeframe: int
            Bar frequency.
        startdate: str (fmt: Y-m-d H:M:s)
            Date that specifies the startpoint.
        enddate: str (fmt: Y-m-d H:M:s)
            Date that specifies the endpoint.

        Returns
        -------
        df: pandas.DataFrame
            Price data (ohlcv).
        """
        if timeframe == 1440:
            url = f"eod/{ticker}.{exchange}?"
            if startdate:
                url = url + f"from={startdate}&"
            if enddate:
                url = url + f"to={enddate}&"
            df = pd.DataFrame.from_dict(self._get_response(url))
        else:
            if not startdate:
                startdate = "2017-01-01 00:00:00"
                enddate = datetime.datetime.now()
            startdate = pd.to_datetime(startdate)
            enddate = pd.to_datetime(enddate)

            intervals = pd.date_range(start=startdate, end=enddate, freq=f"100D")
            intervals = np.unique(np.concatenate([[startdate], intervals.to_list(), [enddate]]))
            intervals = (pd.to_datetime(intervals) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            df = pd.DataFrame()

            for i in range(1, len(intervals)):
                start = intervals[i-1] 
                end = intervals[i]
                url = f"intraday/{ticker}.{exchange}?from={start}&to={end}&interval={timeframe}m&"
                temp_df = pd.DataFrame.from_dict(self._get_response(url))
                df = pd.concat([df, temp_df])       
            
        if "timestamp" in df.columns:
            df.insert(loc=0, column="date", value=pd.to_datetime(df.timestamp, unit="s"))
            df.drop(["timestamp", "gmtoffset"], axis=1, inplace=True)

        df.drop_duplicates(subset=['date'], inplace=True)
        print(df)

        return df

    def populate_tickers(self, types, timeframes):
        """ Downloads price data for every ticker of interest.

        This method downloads all tickers of the specified
        asset type and bar frequency. Stores results in
        ".../PriceData/..." in .csv file. Uses parallelization.

        Parameters
        ----------
        types : list
            List of strings with asset types of interest.
        timeframes: list
            List of integers with bar frequency of interest.

        Returns
        -------
        None
        """
        self._print(msg="Populating Tickers!")

        for typ in types:
            self._create_folder(f"PriceData/{typ}")
            for timeframe in timeframes:
                self._create_folder(f"PriceData/{typ}/{timeframe}")

        tickers = [pl.read_csv(f"{self.path}/TickerInfo/TickerLists/{exchange}.csv").to_pandas()\
                   for exchange in self.exchanges]
        ticker_list = pd.concat(tickers)
        self.duplicates = pd.read_csv("D:/EOD/duplicates.csv")

        if not types:
            for typ in ticker_list.Type.unique():
                self._create_folder(f"PriceData/{typ}") 
                for timeframe in timeframes:
                    self._create_folder(f"PriceData/{typ}/{timeframe}")

        l = []
        selection = ["Code", "Name", "Exchange", "Type", "Isin"]
        for code, _, exchange, t, _ in ticker_list.loc[:, selection].to_numpy():
            if not types or t in types:
                for timeframe in timeframes:
                    l.append((code, exchange, timeframe, t))
            
        with Pool(self.cores) as p:
            p.starmap(self._populate_one_price_data, l)
                                      
        self.get_available_files()

        return None

    def _populate_one_price_data(self, ticker, exchange, timeframe, t):
        """ Helper function to populate one ticker
        
        This method checks if the ticker to download
        is a duplicate and downloads it if it is not for
        every bar frequency of interest.

        Parameters
        ----------
        ticker : str
            Ticker of interest.
        exchange: str
            Exchange from which the ticker can be downloaded.
        timeframes: list
            List of integers with bar frequencies of interest.
        t: str
            Asset type of the symbol.
        types: list
            List of asset types of interest.

        Returns
        -------
        None
        """
        if ticker in self.duplicates.Code.values:
                info = self.duplicates.loc[
                    np.logical_and(self.duplicates.Code     == ticker,
                                   self.duplicates.Exchange == exchange)
                    ]
                if info.Type.values[0] == "None":
                    self._print_ticker_info("not downloaded - DUPLICATE.", t, ticker, "")
        else:
            try:
                if not os.path.isfile(
                        f"{self.path}/PriceData/{t}/{timeframe}/{ticker}_{timeframe}.csv"):
                    df = self.download_ticker(ticker=ticker, 
                                                exchange=exchange, 
                                                timeframe=timeframe)
                    pl.from_pandas(df).to_csv(
                        f"{self.path}/PriceData/{t}/{timeframe}/{ticker}_{timeframe}.csv"
                        )
                    t1 = pd.to_datetime(df.date.iloc[1]).date()
                    t2 = pd.to_datetime(df.date.iloc[-1]).date()
                    self._print_ticker_info(f"from {t1} to {t2} successfully downloaded!",
                                            t, ticker, timeframe)
                else:
                    self._print_ticker_info("already exists!", t, ticker, timeframe)
                    
            except Exception as e:
                self._print_ticker_info(f"failed due to: {e}", t, ticker, timeframe)

        return None

    def populate_fundamentals(self, types, force_update=False):
        """ Downloads fundamental data for every available ticker.

        Parameters
        ----------
        force_update : boolean
            If True, fundamentals will be downloaded regardless
            if it is already available.

        Returns
        -------
        None
        """
        if not os.path.isfile(f"{self.path}/Temp/available_files.csv"):
            self.get_available_files()

        df = pl.read_csv(f"{self.path}/Temp/available_files.csv").to_pandas()
        if types:
            df = df.loc[[typ in types for typ in df.type]]
        iter = df.loc[:,["symbol", "exchange", "type"]].to_numpy()  

        l = []
        for symbol, exchange, typ in iter:
            self._create_folder(folder=f"Fundamentals/{typ}")
            if typ == "ETF":
                l.append((symbol, "XETRA", typ, force_update))
            else:
                l.append((symbol, exchange, typ, force_update))

        with Pool(self.cores) as p:
            p.starmap(self._populate_one_fundamental, l)

        self.get_available_files()
    
        return None

    def _populate_one_fundamental(self, symbol, exchange, typ, force_update):
        """ Helper function that downloads fundamental data 
            for a single ticker."""

        try:
            if not os.path.isfile(f"{self.path}/Fundamentals/{typ}/{symbol}.txt") or force_update:              
                url = f"fundamentals/{symbol}.{exchange}?"
                response = self._get_response(url)

                with open(f"{self.path}/Fundamentals/{typ}/{symbol}.txt", "w") as outfile:
                    json.dump(response, outfile)
                
                self._print_ticker_info("Fundamentals successfully downloaded!", typ, symbol, "")
            else:
                self._print_ticker_info("Fundamentals already exists!", typ, symbol, "")

        except Exception as e:
            self._print_ticker_info(f"failed due to: {e}", typ, symbol, "")
        
    def populate_macroeconomics(self, force_update=False):
        """ Downloads macroeconomical data for every available country.

        Parameters
        ----------
        force_update : boolean
            If True, macroeconomics will be downloaded regardless
            if it is already available.

        Returns
        -------
        None
        """
        self._print(msg=f"Populating Macroeconomics!")
        self._create_folder("/Macroeconomics")

        country_codes = pl.read_csv(f"{self.path}/country_codes.csv").to_pandas()
        macro_indicators = pl.read_csv(f"{self.path}/macroeconomic_indicators.csv").to_pandas()

        l = []
        for code, country in country_codes.to_numpy():
            l.append((code, country, macro_indicators.indicator, force_update))

        with Pool(self.cores) as p:
            p.starmap(self._populate_one_macroeconomics, l)

        return None

    def _populate_one_macroeconomics(self, code, country, macro_indicators, force_update):
        """ Helper function to download all macroeconomical indicators for
            a single country"""

        if os.path.isfile(f"{self.path}/Macroeconomics/{code}.csv") and not force_update:
            self._print(msg=f"Macroeconomics already exist for:  {code}, {country}")
        else:
            res = []
            for macro_indicator in macro_indicators:
                url = f"macro-indicator/{code}?indicator={macro_indicator}&"
                response = self._get_response(url=url)
                temp_df = pd.DataFrame.from_dict(response)
                if temp_df.empty:
                    continue
                temp_df.columns = [col.lower() for col in temp_df.columns]
                temp_df.date = pd.to_datetime(temp_df.date, errors='coerce')
                temp_df = temp_df.loc[temp_df.date > pd.to_datetime("1900-01-01")]
                temp_df.set_index("date", inplace=True)
                temp_df = temp_df.value
                temp_df.name = macro_indicator
                res.append(temp_df)
            
            if len(res) == 0:
                self._print(msg=f"No Macroeconomics available for: {code}, {country}")
                return None

            df = pd.concat(res, axis=1).sort_index(ascending=False) 
            df = df.reset_index()
            pl.from_pandas(df).to_csv(f"{self.path}/Macroeconomics/{code}.csv")
            self._print(msg=f"Macroeconomics downloaded for:  {code}, {country}")

        return None

    def update_price_data(self, asset_types, timeframes):
        """ Updates price data for asset types and bar frequencies
            of interest.

        Checks if the price data for a ticker is out of date and
        updates it if neccessary. Stores price data in ".../PriceData/...".
        Uses parallelization.

        Parameters
        ----------
        asset_types : list
            List of strings that specify asset types of interest.
        timeframes: list
            List of integers that specify bar frequencies of interest.

        Returns
        -------
        None
        """
        self._print(msg=f"Updating files!")
        if not os.path.isfile(f"{self.path}/Temp/available_files.csv"):
            self.get_available_files()
            
        df = pl.read_csv(f"{self.path}/Temp/available_files.csv").to_pandas()
        
        # Filtering for asset types
        if asset_types:
            ind = np.array([df.type.to_numpy() == asset_type 
                            for asset_type in asset_types]).T.sum(axis=1) > 0
            df = df.loc[ind]

        # Filtering for timeframes
        if timeframes: 
            ind = np.array([df.timeframe.to_numpy() == timeframe 
                            for timeframe in timeframes]).T.sum(axis=1) > 0
            df = df.loc[ind]

        selection = ["symbol", "exchange", "type", "timeframe", "path"]
        iter = df.loc[:, selection].to_numpy()

        l = []
        for symbol, exchange, typ, timeframe, path in iter:
            l.append((symbol, exchange, typ, timeframe, path))

        with Pool(self.cores) as p:
            p.starmap(self._update_one_price_data, l)   

        self.get_available_files()   

        return None            
    
    def _update_one_price_data(self, symbol, exchange, typ, timeframe, path):
        """ Helper function that updates the price data
            for a single ticker."""

        try:
            file = pl.read_csv(path).to_pandas()
            last_date = pd.to_datetime(file.date.iloc[-1])
            if last_date < pd.to_datetime(datetime.now().date()):
                update = self.download_ticker(ticker=symbol,
                                            exchange=exchange,
                                            timeframe=timeframe,
                                            startdate=last_date)
                if pd.to_datetime(update.date.iloc[-1]) == last_date:
                    self._print_ticker_info(f"no new data available at {last_date}!",
                                             typ, symbol, timeframe) 
                    return None

                update = pd.concat([file, update])
                update.date = pd.to_datetime(update.date)
                update.drop_duplicates(subset=['date'], keep="last", inplace=True)  
                update.date = update.date.apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                self._print_ticker_info(f"up to date at {last_date}!", typ, symbol, timeframe)
                return None

            date_before = pd.to_datetime(file.date.iloc[-1])
            date_after = update.date.iloc[-1]
            update = update.reset_index(drop=True)  
            pl.from_pandas(update).to_csv(path)   
            self._print_ticker_info(f"updated from {date_before} to {date_after}!",
                                    typ, symbol, timeframe)
                
            return None

        except Exception as e:
            print(e, symbol)
        
        return None

if __name__ == "__main__":
    
    api_key = "60abb74aa4a375.36407570"
    folder = "D:/EOD"
    exchanges = ["F", "XETRA"]
    types = ["ETF"]
    timeframes = [1440]
    eod = EODDownloader(api_key=api_key, exchanges=exchanges, folder=folder)
    # eod.populate_tickers(types=types, timeframes=timeframes)
    # eod.update_price_data(asset_types=types, timeframes=timeframes)
    # eod.populate_fundamentals(types=types, force_update=False)
    # eod.populate_macroeconomics(force_update=False)
    eod.get_available_files()

