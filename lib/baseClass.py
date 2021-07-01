import os
from pandas import to_datetime
from datetime import datetime

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

if __name__ == "__main__":
    pass