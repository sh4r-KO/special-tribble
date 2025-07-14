#https://fxcodebase.com/code/viewtopic.php?f=51&t=73024
#https://fxcodebase.com/bin/forexconnect/1.6.0/python/web-content.html#index.html
#https://fxcodebase.com/bin/forexconnect/1.6.0/python/howto.html
#outdated   :   https://github.com/gehtsoft/forex-connect/tree/master/samples/Python
#python -m pip install numpy
#python -m pip install forexconnect
#python -m pip install pandas
#python3.7

demo_account = "D251064500"
demo_password = "Rabr5"
demo_connection = "Demo"
url = "http://www.fxcorporate.com/Hosts.jsp"

from forexconnect import fxcorepy
from forexconnect import ForexConnect, ResponseListener, Common
from forexconnect import fxcorepy, ForexConnect
 
 
def session_status_changed(session: fxcorepy.O2GSession,
                           status: fxcorepy.AO2GSessionStatus.O2GSessionStatus):
    print("Trading session status: " + str(status))
 
def main():
 
    with ForexConnect() as fx:
        try:
            fx.login(demo_account, demo_password, url,
                     demo_connection, session_status_callback=session_status_changed)
 
        #TBD: your ForexConnect API logic here.


        
 
        except Exception as e:
            print("Exception: " + str(e))
 
        try:
            fx.logout()
        except Exception as e:
            print("Exception: " + str(e))
 
 
if __name__ == "__main__":
    main()