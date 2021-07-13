import os
import csv
from pandas import to_datetime
from datetime import datetime
from numpy import logical_and

class BaseClass:

    def __init__(self, folder, verbose):
        self.verbose = verbose
        if folder:
            self.path = folder
        else:
            self.path = os.path.dirname(os.path.realpath(__file__)) 

    def _print(self, msg):
            if self.verbose:
                print(f"{datetime.now().time().strftime('%H:%M:%S')}:\t{msg}")

    def _print_ticker_info(self, msg, asset_type, symbol, timeframe):
        if self.verbose:
            asset_type += ":"
            ticker = symbol + " " + str(timeframe)
            date = datetime.now().time().strftime('%H:%M:%S')
            print(f"{date:<24}{asset_type:<16}{ticker:<18}{msg:<20}")

    def _create_folder(self, folder):
        if not os.path.exists(self.path + f"/{folder}/"):
            self._print(msg=f"Creating new folder {folder}...")
            os.makedirs(self.path + f"/{folder}/")
    
    def _to_dt_idx(self, df):
        df = df.set_index("date")
        df.index = to_datetime(df.index)
        
        return df

    def _reset_idx(self, df):
        df = df.reset_index()
        df = df.drop(columns="index")

        return df
    
    def _already_tried(self, na_files, ticker, timeframe, verbose=True):
        # Check if data has already been tried to download and failed
        flag = logical_and(
            na_files.iloc[:, 1].values == [ticker],
            na_files.iloc[:, 2].values == [timeframe]
            ) 
        if any(flag) and verbose:
            self._print_ticker_info("NOT available, already tried!", 
                                    "Crypto", ticker, timeframe)
            return True
        return False
    
    def _write_tried(self, asset_type, ticker, timeframe):
        with open(f"{self.path}/Temp/not_available_files.csv", "a+", newline="\n") as f:
            writer = csv.writer(f)
            writer.writerow([asset_type, ticker, timeframe])
            f.close()

if __name__ == "__main__":
    pass