import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import datetime as dt
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import matplotlib.pyplot as plt
import pandas_datareader as web
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.layers import Dense , Dropout , LSTM
#from tensorflow.keras.models import Sequential

st.title('FINANCIAL MODELLING WITH MACHINE LEARNING')

rad=st.sidebar.radio("Navigation",["Home","Stock Forecast App","CryptoCurrency prediction using Machine learning"])

if rad=="Home":
  st.header('Project submission ')
  st.subheader('Radhika --1917631')

if rad=="Stock Forecast App":

  START = "2016-01-01"
  TODAY = date.today().strftime("%Y-%m-%d")
  st.title('Stock Forecast App')
  stocks = ('GOOG', 'AAPL', 'MSFT', 'INFY','FB','AMZN')
  selected_stock = st.selectbox('Select dataset for prediction', stocks)
  n_years = st.slider('Years of prediction:', 1, 4)
  period = n_years * 365
  @st.cache_data
  def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data
  data_load_state = st.text('Loading data...')
  data = load_data(selected_stock)
  data_load_state.text('Loading data... done!')
  
  st.subheader('Raw data')
  st.write(data.tail())
  # Plot raw data
  def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
  plot_raw_data()
  
  # Predict forecast with Prophet.
  df_train = data[['Date','Close']]
  df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
  m = Prophet()
  m.fit(df_train)
  future = m.make_future_dataframe(periods=period)
  forecast = m.predict(future)
  
  # Show and plot forecast
  st.subheader('Forecast data')
  st.write(forecast.tail())
  st.write(f'Forecast plot for {n_years} years')
  fig1 = plot_plotly(m, forecast)
  st.plotly_chart(fig1)
  st.write("Forecast components")
  fig2 = m.plot_components(forecast)
  st.write(fig2)

if rad=="CryptoCurrency prediction using Machine learning":
  from sklearn.svm import SVR
  import matplotlib.pyplot as plt
  import datetime as dt
  plt.style.use('fivethirtyeight')
  START = dt.datetime(2016,1,1)
  END = dt.datetime.now()
  from pandas_datareader import data as pdr
  import yfinance as yf
  st.title('Crypto Currency Price Prediction App ')
  st.subheader('Using Support Vector Machine')
  tick = ('BTC-USD', 'ETH-USD', 'USDT_USD', 'BNB-USD','DOGE-USD')
  ticker = st.sidebar.selectbox('Select dataset for prediction', tick)
  
  st.subheader("Data selected :- " +ticker)
  
  @st.cache_data
  def load_data(ticker):
    data = yf.download(ticker, START, END)
    data.reset_index(inplace=True)
    return data
  
  data_load_state = st.text('Loading data...')
  data = load_data(ticker)
  data_load_state.text('Loading data... done!')
  st.subheader('Real time data')
  st.write(data)
  df=data
  
  future_days=5
  df[str(future_days)+'_Day_Price_Forecast'] = df[['Close']].shift(-future_days)
  df[['Close', str(future_days)+'_Day_Price_Forecast']]
  X = np.array(df[['Close']])
  X= X[:df.shape[0]-future_days]
  y = np.array(df[str(future_days)+'_Day_Price_Forecast'])
  y = y[:-future_days]
  
  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test =train_test_split(X,y,test_size =0.2)
  from sklearn.svm import SVR
  svr_rbf = SVR(kernel='rbf',C= 1e3, gamma=0.00001)
  svr_rbf.fit(x_train,y_train)
  svr_rbf_confidence =svr_rbf.score(x_test , y_test)
  st.write('Accuracy Score:- :',svr_rbf_confidence * 100)
  svm_prediction =svr_rbf.predict(x_test)
  
  st.subheader('Actual VS Predicted prices using SVM')
  fig = plt.figure(figsize=(12,4))
  plt.plot(svm_prediction, label='Prediction',lw=2,alpha=0.7)
  plt.plot(y_test, label='Actual',lw=2,alpha=0.7)
  plt.title('Prediction Vs Actual')
  plt.ylabel('Price in USD')
  plt.xlabel('Time')
  plt.legend()
  plt.xticks(rotation=45)
  st.pyplot(fig)
  st.subheader('Prediction')



if rad=="CryptoCurrency prediction using Deep learning":
  START = dt.datetime(2016,1,1)
  END = dt.datetime.now()
  from pandas_datareader import data as pdr
  import yfinance as yf
  st.title('Crypto Currency Price Prediction App ')
  st.subheader('Using deep learning')
  tick = ('BTC-USD', 'ETH-USD', 'USDT_USD', 'BNB-USD','DOGE-USD')
  ticker = st.sidebar.selectbox('Select dataset for prediction', tick)
  
  st.subheader("Data selected :- " +ticker)

  
  def load_data(ticker):
    data = yf.download(ticker, START, END)
    data.reset_index(inplace=True)
    return data

  data_load_state = st.text('Loading data...')
  data = load_data(ticker)
  data_load_state.text('Loading data... done!')
  
  st.subheader('Real time data')
  st.write(data)
  st.subheader('Real time data -Start')
  st.write(data.head())
  st.subheader('Real time data -End')
  st.write(data.tail())
  
  #Prepare Data
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
  prediction_days = 60
  
  x_train, y_train = [], []
  
  for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])
  
  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
  
  st.write('Creating neural network ')
  #Create neural network
  model = Sequential()
  model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
  model.add(Dropout(0.2))
  model.add(LSTM(units=50, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(units=50))
  model.add(Dropout(0.2))
  model.add(Dense(units=1))
  
  model.compile(optimizer='adam', loss='mean_squared_error')
  model.fit(x_train, y_train, epochs=25, batch_size=32 )
  st.write('Testing Model ')
  
  #Testing the model
  test_start = dt.datetime(2020,1,1)
  test_end = dt.datetime.now()
  
  def load_test_data(ticker):
    test_data = yf.download(ticker, test_start,test_end)
    test_data.reset_index(inplace=True)
    return test_data
  
  data_load_state = st.text('Loading data...')
  test_data = load_test_data(ticker)
  
  actual_prices = test_data['Close'].values
  total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)
  
  model_inputs = total_dataset[len(total_dataset)-len(test_data) - prediction_days:].values
  model_inputs = model_inputs.reshape(-1,1)
  model_inputs = scaler.fit_transform(model_inputs)
  
  x_test = []
  
  for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x,0])
  x_test = np.array(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
  
  prediction_prices = model.predict(x_test)
  prediction_prices = scaler.inverse_transform(prediction_prices)
  real_data = [model_inputs[len(model_inputs) - prediction_days : len(model_inputs)+1, 0]]
  real_data =  np.array(real_data)
  real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
  prediction = model.predict(real_data)
  prediction = scaler.inverse_transform(prediction)
  
  st.subheader('Actual VS Predicted prices using LSTM')
  
  fig = plt.figure(figsize = (10, 5))
  plt.plot(actual_prices, color = 'black', label='Actual Prices')
  plt.plot(prediction_prices, color='green', label ='Predicted Prices')
  plt.title(f'{ticker} price_prediction')
  plt.xlabel('Time')
  plt.ylabel('Price')
  plt.legend(loc='upper left')
  
  st.pyplot(fig)
  st.subheader('Prediction')
  est_val=(f"Tomorrow's {ticker} price: {prediction}")
  st.subheader(est_val)



