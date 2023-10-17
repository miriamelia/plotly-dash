from dash import Dash, html, dcc, callback, Input, Output
import pandas as pd
import dash_ag_grid as dag
import numpy as np
import plotly.express as px

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/Dash-Course/US-Exports/2011_us_ag_exports.csv')

app = Dash(__name__)

app.layout = html.Div([
    html.Div(id='my-title', children="US Agricultural Exports in 2011"),
    html.Br(),
    dcc.Dropdown(df['state'].unique(), ['Alabama', 'Arkansas'], id="state-dropdown", multi=True),
    html.Br(),
    dcc.Graph(id='graph1'),
])

@callback(Output('graph1', 'figure'),
          Input('state-dropdown', 'value'))
def display_state_selected(state_selected):
    df_country = df[df['state'].isin(state_selected)]                 # for one: df.state == state_selected
    fig = px.bar(df_country, x='state', y=['beef','pork','fruits fresh'])
    return fig

if __name__ == "__main__":
    app.run(debug=True)