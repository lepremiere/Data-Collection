import os
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
        """ Check if folder exists, otherwise creating it """

        if not os.path.exists(self.path + f"/{folder}/"):
            self.print(msg=f"Creating new folder {folder}...")
            os.makedirs(self.path + f"/{folder}/")

if __name__ == "__main__":
    pass