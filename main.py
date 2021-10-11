from eod import EODDownloader
if __name__ == "__main__":

    api_key = "60abb74aa4a375.36407570"
    folder = "D:/EOD"
    exchanges = ["F", "XETRA"]
    types = []
    timeframes = [1440]
    eod = EODDownloader(api_key=api_key, exchanges=exchanges, folder=folder)
    for i in [15, 60, 240]:
        df = eod.download_ticker(ticker="GSPC",
                                exchange="INDX", 
                                timeframe=i, 
                                startdate="2010-01-01", 
                                enddate="2022-01-01")
        df.to_csv(f"sp_{i}.csv", index=False)