import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle


app = dash.Dash()
server = app.server

scaler = MinMaxScaler(feature_range=(0, 1))


df_nse = pd.read_csv("./NSE-TATAResult.csv")

df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
df_nse.index = df_nse['Date']


data = df_nse.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Close'])
new_data2 = pd.DataFrame(index=range(0, len(df_nse)), columns=[
                         'Date', 'CloseRateOfChange'])

for i in range(0, len(data)):
    new_data["Date"][i] = data['Date'][i]
    new_data["Close"][i] = data["Close"][i]
    new_data2["Date"][i] = data['Date'][i]
    new_data2["CloseRateOfChange"][i] = data["CloseRateOfChange"][i]

new_data.index = new_data.Date
new_data.drop("Date", axis=1, inplace=True)
new_data2.index = new_data2.Date
new_data2.drop("Date", axis=1, inplace=True)

dataset = new_data.values
dataset2 = new_data2.values

train = dataset[0:987, :]
valid = dataset[987:, :]
train2 = dataset2[0:987, :]
valid2 = dataset2[987:, :]


scaler = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
scaled_data2 = scaler2.fit_transform(dataset2)

model = load_model("saved_model.h5")
model2 = load_model("saved_model_rate_of_change.h5")

inputs = new_data[len(new_data)-len(valid)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
inputs2 = new_data2[len(new_data2)-len(valid2)-60:].values
inputs2 = inputs2.reshape(-1, 1)
inputs2 = scaler2.transform(inputs2)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test2 = []
for i in range(60, inputs2.shape[0]):
    X_test2.append(inputs2[i-60:i, 0])
X_test2 = np.array(X_test2)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
X_test2 = np.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1], 1))
closing_price2 = model2.predict(X_test2)
closing_price2 = scaler2.inverse_transform(closing_price2)

train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price
valid['indexPre'] = valid.index
valid['indexPre'] = pd.DatetimeIndex(valid['indexPre']) + pd.DateOffset(1)
train2 = new_data2[:987]
valid2 = new_data2[987:]
valid2['Predictions'] = closing_price2
valid2['indexPre'] = valid2.index
valid2['indexPre'] = pd.DatetimeIndex(valid2['indexPre']) + pd.DateOffset(1)

# xgboost model


def chuyenDoi(array):
    result = []
    for element in array:
        result.append(element[0])
    result = np.array(result)
    return result


X_test_xgboost = []
X_test_xgboost2 = []
for i in range(1, len(new_data) - 986):
    xxx = []
    for j in range(0, 987):
        xxx.append(new_data.values[i + j])

    xxx = chuyenDoi(xxx)
    X_test_xgboost.append(xxx)
X_test_xgboost = np.array(X_test_xgboost)

for i in range(1, len(new_data2) - 986):
    xxx2 = []
    for j in range(0, 987):
        xxx2.append(new_data2.values[i + j])

    xxx2 = chuyenDoi(xxx2)
    X_test_xgboost2.append(xxx2)
X_test_xgboost2 = np.array(X_test_xgboost2)

loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
xgboost_closing_price = loaded_model.predict(X_test_xgboost)
# predictions = [round(value) for value in xgboost_closing_price]
valid['Predictions-xgboost'] = xgboost_closing_price

loaded_model2 = pickle.load(open("pima.pickle_rate_of_change.dat", "rb"))
xgboost_closing_price2 = loaded_model2.predict(X_test_xgboost2)
# predictions = [round(value) for value in xgboost_closing_price]
valid2['Predictions-xgboost'] = xgboost_closing_price2


df = pd.read_csv("./stock_data.csv")

app.layout = html.Div([

    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[

        dcc.Tab(label='NSE-TATAGLOBAL Stock Data', children=[
            html.Div([
                dcc.Dropdown(
                    id='dropdown-model',
                    options=[
                        {'label': 'XGBoost Model', 'value': 'xgb'},
                        {'label': 'LSTM Model', 'value': 'lstm'}
                    ],
                    # searchable=False,
                    value='lstm',
                    style={"display": "block", "margin-left": "auto",
                           "margin-right": "auto", "width": "60%"}
                ),

                html.H2("Close Price",
                        style={"textAlign": "center"}),
                dcc.Graph(
                    id="close price",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid.index,
                                y=valid["Close"],
                                mode='markers',
                                name='valid'
                            ),
                            go.Scatter(
                                x=valid['indexPre'],
                                y=valid['Predictions'],
                                name='predicted'
                            )

                        ],
                        "layout":go.Layout(
                            title='XGBoost Model',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                ),
                html.H2("Price Rate of Change",
                        style={"textAlign": "center"}),
                dcc.Graph(
                    id="price rate of change",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid2.index,
                                y=valid2["CloseRateOfChange"],
                                mode='markers',
                                name='valid'
                            ),
                            go.Scatter(
                                x=valid2['indexPre'],
                                y=valid2['Predictions'],
                                name='predicted'
                            )

                        ],
                        "layout":go.Layout(
                            title='XGBoost Model',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }

                )
            ])


        ])
        # ,
        # dcc.Tab(label='Facebook Stock Data', children=[
        #     html.Div([
        #         html.H1("Stocks High vs Lows",
        #                 style={'textAlign': 'center'}),

        #         dcc.Dropdown(id='my-dropdown',
        #                      options=[{'label': 'Tesla', 'value': 'TSLA'},
        #                               {'label': 'Apple', 'value': 'AAPL'},
        #                               {'label': 'Facebook', 'value': 'FB'},
        #                               {'label': 'Microsoft', 'value': 'MSFT'}],
        #                      multi=True, value=['FB'],
        #                      style={"display": "block", "margin-left": "auto",
        #                             "margin-right": "auto", "width": "60%"}),
        #         dcc.Graph(id='highlow'),
        #         html.H1("Stocks Market Volume", style={'textAlign': 'center'}),

        #         dcc.Dropdown(id='my-dropdown2',
        #                      options=[{'label': 'Tesla', 'value': 'TSLA'},
        #                               {'label': 'Apple', 'value': 'AAPL'},
        #                               {'label': 'Facebook', 'value': 'FB'},
        #                               {'label': 'Microsoft', 'value': 'MSFT'}],
        #                      multi=True, value=['FB'],
        #                      style={"display": "block", "margin-left": "auto",
        #                             "margin-right": "auto", "width": "60%"}),
        #         dcc.Graph(id='volume')
        #     ], className="container"),
        # ])


    ])
])


@app.callback(Output('close price', 'figure'),
              [Input('dropdown-model', 'value')])
def update_graph(selected_dropdown):
    # print('select dropdown: ', selected_dropdown)
    index = 'Predictions'
    title = 'LSTM Model'
    if(selected_dropdown == 'xgb'):
        index = 'Predictions-xgboost'
        title = 'XGBoost Model'

    figure = {
        "data": [
            go.Scatter(
                x=valid.index,
                y=valid["Close"],
                mode='markers',
                name='valid'
            ),
            go.Scatter(
                x=valid['indexPre'],
                y=valid[index],
                name='predicted'
            )

        ],
        "layout": go.Layout(
            title=title,
            xaxis={'title': 'Date'},
            yaxis={'title': 'Closing Rate'}
        )
    }
    return figure


@app.callback(Output('price rate of change', 'figure'),
              [Input('dropdown-model', 'value')])
def update_graph(selected_dropdown):
    # print('select dropdown: ', selected_dropdown)
    index = 'Predictions'
    title = 'LSTM Model'
    if(selected_dropdown == 'xgb'):
        index = 'Predictions-xgboost'
        title = 'XGBoost Model'

    figure = {
        "data": [
            go.Scatter(
                x=valid2.index,
                y=valid2["CloseRateOfChange"],
                mode='markers',
                name='valid'
            ),
            go.Scatter(
                x=valid2['indexPre'],
                y=valid2[index],
                name='predicted'
            )

        ],
        "layout": go.Layout(
            title=title,
            xaxis={'title': 'Date'},
            yaxis={'title': 'Closing Rate'}
        )
    }
    return figure


# @app.callback(Output('highlow', 'figure'),
#               [Input('my-dropdown', 'value')])
# def update_graph(selected_dropdown):
#     dropdown = {"TSLA": "Tesla", "AAPL": "Apple",
#                 "FB": "Facebook", "MSFT": "Microsoft", }
#     trace1 = []
#     trace2 = []
#     for stock in selected_dropdown:
#         trace1.append(
#             go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                        y=df[df["Stock"] == stock]["High"],
#                        mode='lines', opacity=0.7,
#                        name=f'High {dropdown[stock]}', textposition='bottom center'))
#         trace2.append(
#             go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                        y=df[df["Stock"] == stock]["Low"],
#                        mode='lines', opacity=0.6,
#                        name=f'Low {dropdown[stock]}', textposition='bottom center'))
#     traces = [trace1, trace2]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data,
#               'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
#                                             '#FF7400', '#FFF400', '#FF0056'],
#                                   height=600,
#                                   title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
#                                   xaxis={"title": "Date",
#                                          'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
#                                                                              'step': 'month',
#                                                                              'stepmode': 'backward'},
#                                                                             {'count': 6, 'label': '6M',
#                                                                              'step': 'month',
#                                                                              'stepmode': 'backward'},
#                                                                             {'step': 'all'}])},
#                                          'rangeslider': {'visible': True}, 'type': 'date'},
#                                   yaxis={"title": "Price (USD)"})}
#     return figure


# @app.callback(Output('volume', 'figure'),
#               [Input('my-dropdown2', 'value')])
# def update_graph(selected_dropdown_value):
#     dropdown = {"TSLA": "Tesla", "AAPL": "Apple",
#                 "FB": "Facebook", "MSFT": "Microsoft", }
#     trace1 = []
#     for stock in selected_dropdown_value:
#         trace1.append(
#             go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                        y=df[df["Stock"] == stock]["Volume"],
#                        mode='lines', opacity=0.7,
#                        name=f'Volume {dropdown[stock]}', textposition='bottom center'))
#     traces = [trace1]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data,
#               'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
#                                             '#FF7400', '#FFF400', '#FF0056'],
#                                   height=600,
#                                   title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
#                                   xaxis={"title": "Date",
#                                          'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
#                                                                              'step': 'month',
#                                                                              'stepmode': 'backward'},
#                                                                             {'count': 6, 'label': '6M',
#                                                                              'step': 'month',
#                                                                              'stepmode': 'backward'},
#                                                                             {'step': 'all'}])},
#                                          'rangeslider': {'visible': True}, 'type': 'date'},
#                                   yaxis={"title": "Transactions Volume"})}
#     return figure


if __name__ == '__main__':
    app.run_server(debug=True)
