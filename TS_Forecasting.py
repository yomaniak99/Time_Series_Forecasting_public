from fbprophet import Prophet
from alpha_vantage.timeseries import TimeSeries

def getData():

    key = 'KGZW12R6CERW119E' # Your key here
    ts = TimeSeries(key, output_format='pandas')

    # data is a pandas dataframe, meta_data is a dict
    data, meta_data = ts.get_daily(symbol='IBM', outputsize= 'full')

    return data.reset_index()

#Initialize our dataframe
def setupDataFrame(data):

    #Converting to make compatible with prophet
    history_close = data[['date', '4. close']]
    history_close = history_close.rename(columns={"date": "ds", "4. close": "y"})

    # setting the logistic growth
    history_close['cap'] = history_close['y'].max()
    history_close['floor'] = history_close['y'].min()

    return history_close

data = getData()
print(data.tail(5))

history_close = setupDataFrame(data)
print("Valeur max de y: " + str(history_close['y'].max()))
print(history_close.tail(5))

# instantiate the model and set parameters
model = Prophet(
    #interval_width=0.95,
    growth='logistic',
    #daily_seasonality=False,
    #weekly_seasonality=False,
    #yearly_seasonality=False,
    #seasonality_mode='multiplicative'
    )

# fit the model to historical data
model.fit(history_close)

#ask to predict x days in advance
future_close = model.make_future_dataframe(
    periods= 365, #predict over 365 days!!
    freq='d',
    include_history=True
)
future_close['cap'] = history_close['y'].max()#history_close['cap']
future_close['floor'] = history_close['y'].min()#history_close['floor']

# predict over the dataset
forecast_close = model.predict(future_close)
print(forecast_close[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

#Ploting a graph
predict_fig = model.plot(forecast_close, xlabel='date', ylabel='close')
predict_fig.savefig('img/close.png')

#see the forecast components
fig2 = model.plot_components(forecast_close)
predict_fig.savefig('img/close_forcast.png')

#interactive figure
#fig = plot_plotly(model, forecast_close)  # This returns a plotly Figure
#py.iplot(fig)

