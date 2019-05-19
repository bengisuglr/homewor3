from math import sqrt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, Holt #,ExponentialSmoothing,  
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

# The following lines are to suppress warning messages.
import warnings
warnings.filterwarnings("ignore")

#Functions needed

def decompmod(name,f,mod='Additive'):
    array = name
    result = sm.tsa.seasonal_decompose(array,freq=f,model=mod,two_sided=False)
    # Additive model means y(t) = Level + Trend + Seasonality + Noise
    result.plot()
    plt.show() # Uncomment to reshow plot, saved as Figure 1.
    return result

def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=24).mean()
    rolstd = pd.Series(timeseries).rolling(window=24).std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    #Perform Dickey-Fuller test:
    print("Results of Dickey-Fuller Test:")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array,copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


def estimate_ses(testarray, alpha):
    numbers = testarray
    estimate = SimpleExpSmoothing(numbers).fit(smoothing_level=alpha,optimized=False).forecast(1)
    return estimate



def estimate_holt(array, alpha, slope, a):
    numbers = array
    model = Holt(numbers)
    fit = model.fit(alpha,slope)
    estimate = fit.forecast(a)[-1]
    return estimate
   
def RMSE(value, estimate):
    ms = mean_squared_error(value, estimate, sample_weight=None, multioutput="uniform_average")
    rmse = sqrt(ms)
    return rmse

def Error_output(estimate1, estimate2, value1, value2):
    estimate1557=[]
    estimate1558=[]
    estimate1557.append(estimate1)
    estimate1558.append(estimate2)

    Error_1557 = RMSE(value1, estimate1557)
    Error_1558 = RMSE(value2, estimate1558)
    
    return Error_1557, Error_1558

def Sum_error(error1, error2):
    totalerror = error1 + error2
    return totalerror

tradedata = pd.read_csv("trade.txt", sep = "\t")
seriesname = "Close"
tradedata["period"] = tradedata["Day"].map(str) + tradedata["Time"]
tradedata.set_index("period")
series = tradedata[seriesname]


def get_small(freq, a, b):
    shortdata = []
    for i in range (0, freq):
        k = a+(i*60)
        shortdata.append(series[k])
        j = b+(i*60)
        shortdata.append(series[j]) 
        i += 1
    return shortdata

#2)a-
#stationarity testing for complete data
print("2)a-")
print("stationarity testing for :59 :00 data:")
print("\n")
endhourseries = get_small(183, 59, 60)
test_stationarity(endhourseries)
result = decompmod(endhourseries, f=24)
test_stationarity(result.trend)
print("result: the data for :59 and :00 values only is not stationary according to dickey-fuller test")
print("\n")
print("stationarity testing for :57 :58 data:")
print("\n")
endhourseries2 = get_small(183, 57, 58)
test_stationarity(endhourseries2)
result = decompmod(endhourseries2, f=24)
test_stationarity(result.trend)
print("result: the data for :57 and :58 values only is not stationary according to dickey-fuller test")
print("\n")

#2)-b
#estimation for 15:57 and 15:58 for 6th of may
#method 1: SES estimation
print("\n")
print("2)b-")
print("only the :57 and :58 time values are used for simplicity")
shortarray= endhourseries2[:-2]
shortarray2= endhourseries2[:-1]
# Function for Simple Exponential Smoothing
alpha = 1   
ses_06051557 = round(estimate_ses(shortarray, alpha)[0], 4)
print ("Simple Exponential Smoothing estimation for 15:57:", ses_06051557)
print("\n")
ses_06051558 = round (estimate_ses(shortarray2, alpha)[0], 4)
print ("Simple Exponential Smoothing estimation for 15:58:", ses_06051558)
print("\n")

#method 2: # Trend estimation with Holt
alpha = 0.04
slope = 0.3
forecast = 1
#alpha and slope adjusted to find the closest values possible
holtfor1557 = round(estimate_holt(shortarray,alpha, slope, forecast),4)
print("Holt trend estimation with alpha for 15:57 =", alpha, ", and slope =", slope, ": ", holtfor1557)
print("\n")
holtfor1558 = round(estimate_holt(shortarray2,alpha, slope, forecast),4)
print("Holt trend estimation with alpha for 15:58 =", alpha, ", and slope =", slope, ": ", holtfor1558)
print("\n")

#chechking for RMSE

series = tradedata[seriesname]
sarray = np.asarray(series)


value1557=[]
value1558=[]
value15_57= sarray[-2]
value15_58= sarray[-1]
value1557.append(value15_57)
value1558.append(value15_58)



#error for SES
Error_ses1557, Error_ses1558 = Error_output(ses_06051557, ses_06051558, value1557, value1558)
print("RMSE for SES estimation for 15:57 and 15:58 in order are;", Error_ses1557,",", Error_ses1558)
print("\n")
#error for holt
Error_holt1557, Error_holt1558 = Error_output(holtfor1557, holtfor1558, value1557, value1558)
print("RMSE for holt estimation for 15:57 and 15:58 in order are;", Error_holt1557,",", Error_holt1558)
print("\n")

total_error_for_ses = Sum_error(Error_ses1557,Error_ses1558)
total_error_for_holt = Sum_error(Error_holt1557,Error_holt1558)

print("error in ses and error in holt in order are", total_error_for_ses, total_error_for_holt)
print("\n")

print("\n")
print("1)c-")
#2)c-
# Load data
tradedata = pd.read_csv("trade.txt", sep = "\t")
tradedata["period"] = tradedata["Day"].map(str) + tradedata["Time"]
tradedata.set_index("period")
tradedata["Volume"].replace(['0', '0.0'], '', inplace=True) # discharge zero
tradedata = tradedata.fillna(method = "ffill")
tradedata = tradedata.fillna(method = "bfill")
seriesname = "Close"

#stationarity testing for complete data
print("a")
print("stationarity testing for :59 :00 data:")
print("\n")
endhourseries = get_small(183, 59, 60)
test_stationarity(endhourseries)
result = decompmod(endhourseries, f=24)
test_stationarity(result.trend)
print("result: the data for :59 and :00 values only is stationary according to plot")
print("\n")
print("stationarity testing for :57 :58 data:")
print("\n")
endhourseries2 = get_small(183, 57, 58)
test_stationarity(endhourseries2)
result = decompmod(endhourseries2, f=24)
test_stationarity(result.trend)
print("result: the data for :57 and :58 values only is stationary according to plot")
print("\n")

#b
#estimation for 15:57 and 15:58 for 6th of may
#method 1: SES estimation
print("\n")
print("b")
print("only the :57 and :58 time values are used for simplicity")
shortarray= endhourseries2[:-2]
shortarray2= endhourseries2[:-1]
# Function for Simple Exponential Smoothing
alpha = 1   
ses_06051557 = round(estimate_ses(shortarray, alpha)[0], 4)
print ("Simple Exponential Smoothing estimation for 15:57:", ses_06051557)
print("\n")
ses_06051558 = round (estimate_ses(shortarray2, alpha)[0], 4)
print ("Simple Exponential Smoothing estimation for 15:58:", ses_06051558)
print("\n")

#method 2: # Trend estimation with Holt
alpha = 0.04
slope = 0.3
forecast = 1
#alpha and slope adjusted to find the closest values possible
holtfor1557 = round(estimate_holt(shortarray,alpha, slope, forecast),4)
print("Holt trend estimation with alpha for 15:57 =", alpha, ", and slope =", slope, ": ", holtfor1557)
print("\n")
holtfor1558 = round(estimate_holt(shortarray2,alpha, slope, forecast),4)
print("Holt trend estimation with alpha for 15:58 =", alpha, ", and slope =", slope, ": ", holtfor1558)
print("\n")

#chechking for RMSE

series = tradedata[seriesname]
sarray = np.asarray(series)


value1557=[]
value1558=[]
value15_57= sarray[-2]
value15_58= sarray[-1]
value1557.append(value15_57)
value1558.append(value15_58)



#error for SES
Error_ses1557, Error_ses1558 = Error_output(ses_06051557, ses_06051558, value1557, value1558)
print("RMSE for SES estimation for 15:57 and 15:58 in order are;", Error_ses1557,",", Error_ses1558)
print("\n")
#error for holt
Error_holt1557, Error_holt1558 = Error_output(holtfor1557, holtfor1558, value1557, value1558)
print("RMSE for holt estimation for 15:57 and 15:58 in order are;", Error_holt1557,",", Error_holt1558)
print("\n")

total_error_for_ses = Sum_error(Error_ses1557,Error_ses1558)
total_error_for_holt = Sum_error(Error_holt1557,Error_holt1558)

print("error in ses and error in holt in order are", total_error_for_ses, total_error_for_holt)
print("\n")

