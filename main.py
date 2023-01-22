import streamlit as st
from datetime import datetime,date
import yfinance as yf
from fbprophet import Prophet
from plotly import graph_objs as go
start='2015-01-01'
end=date.today()
st.title("Prediction app")
stocks=("AAPL","GOOG","MSTF","GME")
selected_stock=st.selectbox("select stocks",stocks)
n_years=st.slider("Years",1,4)
periods=n_years*365

def load_data(ticker):
    data=yf.download(ticker,start,end)
    data.reset_index(inplace=True)
    return data

data_load_states=st.text("Load data ...")
data=load_data(selected_stock)
data_load_states.text("loading data")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"],y=data['Open'],name='Stock open'))
    fig.add_trace(go.Scatter(x=data["Date"],y=data['Close'],name='Stock close'))
    fig.layout.update(title_text='Timeseries',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    st.plotly_chart(fig)

plot_raw_data()

## Forecasting
df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})
m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=periods)
forecast=m.predict(future)
st.subheader('Forecast Data')
st.write(forecast.tail())
st.write('Forecast data')
fig1=plt_plotly(m,forecast)
st.plotly_chart(fig1)
st.write('forecast component')
fig2=m.plot_component(forecast)
st.write(fig2)
