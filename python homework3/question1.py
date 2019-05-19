
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
def decomp(frame,name,f,mod='Additive'):
    series = frame[name]
    array = np.asarray(series, dtype=float)
    result = sm.tsa.seasonal_decompose(array,freq=f,model=mod,two_sided=False)
    # Additive model means y(t) = Level + Trend + Seasonality + Noise
    result.plot()
    plt.show() # Uncomment to reshow plot, saved as Figure 1.
    return result

def decomp0605(name,f,mod='Additive'):
    array = name
    result = sm.tsa.seasonal_decompose(array,freq=f,model=mod,two_sided=False)
    # Additive model means y(t) = Level + Trend + Seasonality + Noise
    result.plot()
    plt.show() # Uncomment to reshow plot, saved as Figure 1.
    return result

def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=60).mean()
    rolstd = pd.Series(timeseries).rolling(window=60).std()
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

def test_train(i1, i2, i3):
    series = tradedata[seriesname]
    sarray = np.asarray(series)
    size = len(series)
    train = series[i1:i2]
    trainarray= np.asarray(train)
    test = series[i2:i3]
    testarray = np.asarray(test)
    return testarray

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
#tradedata["Volume"].replace(['0', '0.0'], '', inplace=True) # discharge zero
#tradedata = tradedata.fillna(method = "ffill")
#tradedata = tradedata.fillna(method = "bfill")


#1)a-
#stationarity testing for complete data
print("1)a-")
print("stationarity testing for complete data")
print("\n")
series = tradedata[seriesname]
test_stationarity(series)
result = decomp(tradedata,seriesname,f=1440)
test_stationarity(result.trend)

print("result: all of the data is not stationary according to dickey-fuller test")
print("\n")
testarray = test_train(0, 10080, len(series) )


#stationarity testing for 6th of may data
print("stationarity testing for 6th of may data")
print("\n")
test_stationarity(testarray)
result = decomp0605(testarray,f=360)
test_stationarity(result.trend)
print("result: 6th of may data is not stationary according to dickey-fuller test")
#1)-b
#estimation for 15:57 and 15:58 for 6th of may
#method 1: SES estimation
print("\n")
print("1)b-")
print("last five days of data is used for simplicity to forecast")
testarray1 = test_train(0, 4320, -2)
testarray2 = test_train(0, 4320, -1)

# Function for Simple Exponential Smoothing
alpha = 0.025   
ses_06051557 = round(estimate_ses(testarray1, alpha)[0], 4)
print ("Simple Exponential Smoothing estimation for 15:57:", ses_06051557)
print("\n")
ses_06051558 = round (estimate_ses(testarray2, alpha)[0], 4)
print ("Simple Exponential Smoothing estimation for 15:58:", ses_06051558)
print("\n")

#method 2: # Trend estimation with Holt
alpha = 0.038
slope = 0.1
forecast = 1
#alpha and slope adjusted to find the closest values possible
holtfor1557 = round(estimate_holt(testarray1,alpha, slope, forecast),4)
print("Holt trend estimation with alpha for 15:57 =", alpha, ", and slope =", slope, ": ", holtfor1557)
print("\n")
holtfor1558 = round(estimate_holt(testarray2,alpha, slope, forecast),4)
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

if total_error_for_ses < total_error_for_holt:
    print ("use ses for future forecast")
else:
    print("use holt for future forecast")
 
print("\n")
# estimating 15:59
testarray3= test_train(0, 4320, len(series))
alpha = 0.038
slope = 0.1
forecast= 1
holtfor1559 = round(estimate_holt(testarray3,alpha, slope, forecast),4)
print("Holt trend estimation for 15:59 with alpha=", alpha, ", and slope =", slope, ": ", holtfor1559)
print("\n")
forecast= 2
holtfor1600 = round(estimate_holt(testarray3,alpha,slope, forecast), 4)
print("Holt trend estimation for 16:00 with alpha=", alpha, ", and slope =", slope, ": ", holtfor1600)
print("\n")
forecast= 1440
holtfor071600 = round(estimate_holt(testarray3,alpha,slope, forecast), 4)
print("Holt trend estimation for 07.05.2019 16:00 with alpha=", alpha, ", and slope =", slope, ": ", holtfor071600)
print("\n")
print("1)c-")

# Load data
tradedata = pd.read_csv("trade.txt", sep = "\t")
seriesname = "Close"
tradedata["period"] = tradedata["Day"].map(str) + tradedata["Time"]
tradedata.set_index("period")
tradedata["Volume"].replace(['0', '0.0'], '', inplace=True) # discharge zero
tradedata = tradedata.fillna(method = "ffill")
tradedata = tradedata.fillna(method = "bfill")

#a
#stationarity testing for complete data
print("stationarity testing for complete data with discarding volume 0")
print("\n")
series = tradedata[seriesname]
test_stationarity(series)
result = decomp(tradedata,seriesname,f=1440)
test_stationarity(result.trend)
print("result: all of the data is not stationary according to plot")
print("\n")
testarray = test_train(0, 10080, len(series) )

#stationarity testing for 6th of may data
print("stationarity testing for 6th of may data with discarding volume 0")
print("\n")
test_stationarity(testarray)
result = decomp0605(testarray,f=360)
test_stationarity(result.trend)
print("result: 6th of May the data is not stationary according to plot")
#b
#estimation for 15:57 and 15:58 for 6th of may
#method 1: SES estimation
print("\n")
print("last five days of data is used for simplicity to forecast")
testarray1 = test_train(0, 4320, -2)
testarray2 = test_train(0, 4320, -1)

# Function for Simple Exponential Smoothing
alpha = 0.025   
ses_06051557 = round(estimate_ses(testarray1, alpha)[0], 4)
print ("Simple Exponential Smoothing estimation for 15:57:", ses_06051557)
print("\n")
ses_06051558 = round (estimate_ses(testarray2, alpha)[0], 4)
print ("Simple Exponential Smoothing estimation for 15:58:", ses_06051558)
print("\n")

#method 2: # Trend estimation with Holt
alpha = 0.038
slope = 0.1
forecast = 1
#alpha and slope adjusted to find the closest values possible
holtfor1557 = round(estimate_holt(testarray1,alpha, slope, forecast),4)
print("Holt trend estimation with alpha for 15:57 =", alpha, ", and slope =", slope, ": ", holtfor1557)
print("\n")
holtfor1558 = round(estimate_holt(testarray2,alpha, slope, forecast),4)
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

if total_error_for_ses < total_error_for_holt:
    print ("use ses for future forecast")
else:
    print("use holt for future forecast")
 
print("\n")
# estimating 15:59
testarray3= test_train(0, 4320, len(series))
alpha = 0.038
slope = 0.1
forecast= 1
holtfor1559 = round(estimate_holt(testarray3,alpha, slope, forecast),4)
print("Holt trend estimation for 15:59 with alpha=", alpha, ", and slope =", slope, ": ", holtfor1559)
print("\n")
forecast= 2
holtfor1600 = round(estimate_holt(testarray3,alpha,slope, forecast), 4)
print("Holt trend estimation for 16:00 with alpha=", alpha, ", and slope =", slope, ": ", holtfor1600)
print("\n")
forecast= 1440
holtfor071600 = round(estimate_holt(testarray3,alpha,slope, forecast), 4)
print("Holt trend estimation for 07.05.2019 16:00 with alpha=", alpha, ", and slope =", slope, ": ", holtfor071600)
print("\n")