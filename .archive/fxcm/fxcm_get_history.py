#https://fxcodebase.com/code/viewtopic.php?f=51&t=73024
#https://fxcodebase.com/bin/forexconnect/1.6.0/python/web-content.html#index.html
#https://fxcodebase.com/bin/forexconnect/1.6.0/python/howto.html
#https://github.com/gehtsoft/forex-connect/tree/master/samples/Python
#https://fxcm-api.readthedocs.io/en/latest/fxcmpy.html#demo-account
#python -m pip install numpy
#python -m pip install forexconnect
#python -m pip install pandas
#pip install matplotlib
#pip install timeframe

demo_account = "D251064500"
demo_password = "Rabr5"
demo_connection = "Demo"
url = "http://www.fxcorporate.com/Hosts.jsp"

from forexconnect import fxcorepy
from forexconnect import ForexConnect, ResponseListener, Common
from forexconnect import fxcorepy, ForexConnect
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

 
def session_status_changed(session: fxcorepy.O2GSession,
                           status: fxcorepy.AO2GSessionStatus.O2GSessionStatus):
    print("Trading session status: " + str(status))
 
def main():
    with ForexConnect() as fx:
        try:
            fx.login(demo_account, demo_password, url,
                     demo_connection, session_status_callback=session_status_changed)

            instrument = "EUR/USD"
            timeframe = "t1"  # https://fxcodebase.com/bin/forexconnect/1.6.0/python/whatisTF.html#:~:text=Meaning-,Example,-(s)

            date_from =  dt.datetime(2018, 6, 25,12)
            date_to =  dt.datetime(2018, 6, 25,12,15)
            quotes_count = 1000  # You can adjust this
            candle_open_price_mode = fxcorepy.O2GCandleOpenPriceMode.PREVIOUS_CLOSE

            history = fx.get_history(instrument, timeframe, date_from, date_to, quotes_count, candle_open_price_mode)
            #print("History data retrieved successfully.")
            print(history)
            print("history legnth : ",len(history))
            df1 = pd.DataFrame(history)
            print(df1.info())
            print(df1.head())
            sub = df1
            
            sub['Date'] = pd.to_datetime(sub['Date'], unit='ns')
            sub.set_index('Date', inplace=True)
            sub['Mid'] = sub[['Bid', 'Ask']].mean(axis=1)
            sub['SMA'] = sub['Mid'].rolling(100).mean()  
            sub[['Mid', 'SMA']].plot(figsize=(10, 6), lw=0.75)
            plt.title(f"{instrument} {timeframe} History")
            plt.xlabel("Time")

            plt.ylabel("Price")
            
            plt.grid()
            plt.legend(["Mid Price", "SMA"])
            plt.tight_layout()

            
            fx.logout()


            plt.show()
        except Exception as e:
            print("Exception: " + str(e))


 
if __name__ == "__main__":
    main()