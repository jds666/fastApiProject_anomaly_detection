# dashapp.py
import dash
from dash.dependencies import Input, Output
from dash import dcc, html
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

def create_dash_app(requests_pathname_prefix: str = None) -> dash.Dash:
    app = dash.Dash(__name__, requests_pathname_prefix=requests_pathname_prefix)

    app.layout = html.Div([
        html.H4('Simple stock plot with adjustable axis'),
        html.Button("Switch Axis", n_clicks=0,
                    id='button'),
        dcc.Graph(id="graph"),
    ])

    @app.callback(
        Output("graph", "figure"),
        Input("button", "n_clicks"))
    def update_graph(selected_dropdown_value):
        dff = df[df['Stock'] == selected_dropdown_value]
        return {
            'data': [{
                'x': dff.Date,
                'y': dff.Close,
                'line': {
                    'width': 3,
                    'shape': 'spline'
                }
            }],
            'layout': {
                'margin': {
                    'l': 30,
                    'r': 20,
                    'b': 30,
                    't': 20
                }
            }
        }

    return app


'''
def create_dash_app(requests_pathname_prefix: str = None) -> dash.Dash:
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/hello-world-stock.csv')

    app = dash.Dash(__name__, requests_pathname_prefix=requests_pathname_prefix)
    app.scripts.config.serve_locally = False
    dcc._js_dist[0]['external_url'] = 'https://cdn.plot.ly/plotly-basic-latest.min.js'

    app.layout = html.Div([
        html.H1('Stock Tickers'),
        dcc.Dropdown(
            id='my-dropdown',
            options=[
                {'label': 'Tesla', 'value': 'TSLA'},
                {'label': 'Apple', 'value': 'AAPL'},
                {'label': 'Coke', 'value': 'COKE'}
            ],
            value='TSLA'
        ),
        dcc.Graph(id='my-graph')
    ], className="container")

    @app.callback(Output('my-graph', 'figure'),
                  [Input('my-dropdown', 'value')])
    def update_graph(selected_dropdown_value):
        dff = df[df['Stock'] == selected_dropdown_value]
        return {
            'data': [{
                'x': dff.Date,
                'y': dff.Close,
                'line': {
                    'width': 3,
                    'shape': 'spline'
                }
            }],
            'layout': {
                'margin': {
                    'l': 30,
                    'r': 20,
                    'b': 30,
                    't': 20
                }
            }
        }
    return app

from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)

app.layout = html.Div([
    html.H4('Simple stock plot with adjustable axis'),
    html.Button("Switch Axis", n_clicks=0, 
                id='button'),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"), 
    Input("button", "n_clicks"))
def display_graph(n_clicks):
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv') # replace with your own data source

    if n_clicks % 2 == 0:
        x, y = 'AAPL_x', 'AAPL_y'
    else:
        x, y = 'AAPL_y', 'AAPL_x'

    fig = px.line(df, x=x, y=y)    
    return fig


app.run_server(debug=True)
'''