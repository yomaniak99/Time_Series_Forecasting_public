from fbprophet import Prophet
from alpha_vantage.timeseries import TimeSeries
from fbprophet.plot import add_changepoints_to_plot
import pandas as pd

import time

def getData(symbol):

    key = 'KGZW12R6CERW119E' # Your key here
    ts = TimeSeries(key, output_format='pandas')

    # data is a pandas dataframe, meta_data is a dict
    data, meta_data = ts.get_daily(symbol=symbol, outputsize= 'full')#full or compact

    return data.reset_index()

#Initialize our dataframe
def setupDataFrame(data):

    #Converting to make compatible with prophet
    history_close = data[['date', '4. close']]
    history_close = history_close.rename(columns={"date": "ds", "4. close": "y"})

    # setting the logistic growth
    history_close['cap'] = history_close['y'].max()
    history_close['floor'] = history_close['y'].min()

    #Creating custom seasonal change based on stock market observation by humans
    history_close['high_season'] = history_close['ds'].apply(January_high)
    history_close['low_season'] = ~history_close['ds'].apply(January_high)

    return history_close

def January_high(ds):
    date = pd.to_datetime(ds)
    return (date.month >= 1 and date.month <= 3)


#Start
Symbol = 'IBM'
data = getData(Symbol)
print(data.tail(5))

history_close = setupDataFrame(data)
print("Valeur max de y: " + str(history_close['y'].max()))
print(history_close.tail(5))

# instantiate the model and set parameters
model = Prophet(
    #interval_width=0.95,
    growth='logistic',
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=True,
    #seasonality_mode='multiplicative',###Maybe????
    interval_width=0.85,#default=80%
    changepoint_range=0.9,
    n_changepoints= 30,#max nb of potential changepoints(red mark)
    changepoint_prior_scale= 0.1,#felxibility of prediction(default=0.05)
    )

#Adding our own seasonality to the model
#model.add_seasonality(name='monthly', period=30.5, fourier_order=5) #Adds monthly seasonality
model.add_seasonality(name='monthly_high_season', period=4, fourier_order=5, condition_name='high_season')
model.add_seasonality(name='monthly_low_season', period=8, fourier_order=5, condition_name='low_season')

# fit the model to historical data
model.fit(history_close)

#ask to predict x days in advance
future_close = model.make_future_dataframe(
    periods= 1000, #predict over 1000 days!!
    freq='d',
    include_history=True
)
#Taking our custom modification to the model in consideration in the predicted model
future_close['cap'] = history_close['y'].max()
future_close['floor'] = history_close['y'].min()
future_close['high_season'] = future_close['ds'].apply(January_high)
future_close['low_season'] = ~future_close['ds'].apply(January_high)

# predict over the dataset
forecast_close = model.predict(future_close)
print(forecast_close[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

#Ploting a graph
predict_fig = model.plot(forecast_close, xlabel='date', ylabel='close')
predict_fig.gca().set_title(Symbol, size=14)
#adds potential changepoints, for detecting abrupt change in the datas
#Could be set manually with Prophet(changepoints=['2014-01-01'])
add_changepoints_to_plot(predict_fig.gca(), model, forecast_close)
predict_fig.savefig('img/close.png')
#predict_fig.show()

#see the forecast components
fig2 = model.plot_components(forecast_close)
predict_fig.savefig('img/close_forcast.png')

#interactive figure
#fig = plot_plotly(model, forecast_close)  # This returns a plotly Figure
#py.iplot(fig)

