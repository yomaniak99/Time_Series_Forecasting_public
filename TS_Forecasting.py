import pandas as pd
from fbprophet import Prophet

#read csv file
full_csv = pd.read_csv('datas/daily_IBM.csv', header = 0)
print(full_csv.head(5))

#Initialize our dataframe
history_close = full_csv[['timestamp', 'close']]
history_close = history_close.rename(columns={"timestamp": "ds", "close": "y"})

#setting the logistic growth
history_close['cap'] = history_close['y'].max()#full_csv[['high']]
history_close['floor'] = history_close['y'].min()#full_csv[['low']]
print("Valeur max de y: " + str(history_close['y'].max()))
print(history_close.head(5))

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

#ask to predict x days in advance<
future_close = model.make_future_dataframe(
    periods=31, #predict over 31 days!!
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