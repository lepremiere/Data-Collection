from eod import EODDownloader
if __name__ == "__main__":

    api_key = "60abb74aa4a375.36407570"
    folder = "D:/EOD"
    exchanges = ["F", "XETRA"]
    types = []
    timeframes = [1440]
    eod = EODDownloader(api_key=api_key, exchanges=exchanges, folder=folder)
    df = eod.download_ticker(ticker="GSPC", exchange="INDX", timeframe=1440, startdate="1900-01-01", enddate="2022-01-01")
    df.to_csv("SP500_D1.csv")