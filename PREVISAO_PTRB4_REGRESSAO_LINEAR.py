import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import plotly
plotly.offline.init_notebook_mode()
import datetime

dataset = pd.read_csv('petr4_1_2010_11_2017.csv')

dataset['Date'] = pd.to_datetime(dataset['Date'])
x1 = dataset.Date
y1 = dataset.Close
data = [go.Scatter(x=x1, y=y1)]
layout = go.Layout(
    xaxis = dict(
        range = ['01-01-2010','11-04-2017'],
        title='Ano'
    ),
    yaxis = dict(
        range = [min(x1), max(y1)],
        title = 'Valor da Acao'
    ))
fig = go.Figure(data = data, layout = layout)
py.plot(fig)
#py.offline.iplot(fig)

dataset2 = dataset.head(7)
dados = go.Candlestick(x = dataset2.Date, 
                       open = dataset2.Open,
                       high = dataset2.High,
                       low = dataset2.Low,
                       close = dataset2.Close,
                      )
data = [dados]
py.offline.iplot(data, filename = 'grafico_candlestick')

treino = dataset

x = treino.High[:100]
y = treino.Close[:100]
plt.scatter(x, y, color = 'r')
plt.xlabel('preco maxima')
plt.ylabel('vlr fechamento')
plt.axis([min(x), max(x), min(y), max(y)])
plt.autoscale('false')
plt.show()

features = ['Open', 'High', 'Low', 'Volume']
treino = treino[features]

y = dataset['Close']

X_treino, X_teste, Y_treino, Y_teste = train_test_split(treino, y, random_state=42)

lr_model = LinearRegression()

lr_model.fit(X_treino, Y_treino)

lr_model.coef_

lr_model.predict(X_teste)[:10]

Y_teste[:10]

RMSE = mean_squared_error(Y_teste, lr_model.predict(X_teste))**0.5
RMSE