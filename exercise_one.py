from dash import Dash, html, dcc, callback, Input, Output
import pandas as pd
import dash_ag_grid as dag
import numpy as np
import plotly.express as px

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/Dash-Course/makeup-shades/shades.csv')

appA = Dash(__name__)
appB = Dash(__name__)


makeup_scatter_plot = px.scatter(df, x="V", y="S", color="brand", hover_data="hex")

# create a basic dropdown, provide options and a value to dcc.Dropdown in that order
appA.layout = html.Div([
    dcc.Dropdown(list(set(df['brand'].to_list())), 'Revlon', id="dropdown_1"),
    html.Br(),
    # dcc.RadioItems(list(set(df['group'].to_list())), '0', id="radioitems_1")
    dcc.RadioItems(
        options=[
        {'label': 'Fenty Beauty\'s PRO FILT\'R Foundation Only', 'value': '0'},
        {'label': 'Make Up For Ever\'s Ultra HD Foundation Only', 'value': '1'},
        {'label': 'US Best Sellers', 'value': '2'},
        {'label': 'BIPOC-recommended Brands with BIPOC Founders', 'value': '3'},
        {'label': 'BIPOC-recommended Brands with White Founders', 'value': '4'},
        {'label': 'Nigerian Best Sellers', 'value': '5'},
        {'label': 'Japanese Best Sellers', 'value': '6'},
        {'label': 'Indian Best Sellers', 'value': '7'},
        ],
        value='0', id="radioitems_1"
    ),
    dcc.ConfirmDialog(id='confirm-order',
        message='You\'ve selected one of US Best Sellers',
    ),
    html.Br(),
    dcc.Graph(figure=makeup_scatter_plot)
])

@callback(Output('confirm-order', 'displayed'),
          Input('dropdown_1', 'value'))
def display_confirm(value):
    if value == 'Maybelline':
        return True
    return False

# When you use to_dict("records"), it will convert the DataFrame into a list of dictionaries where each dictionary represents a row in the 
# DataFrame, and the keys of the dictionaries correspond to the column names, making it a record-oriented representation of your data.
grid = dag.AgGrid(
    id="exercise-1-b",
    rowData=df.to_dict("records"),
    columnDefs=[{"field": i} for i in df.columns],
    dashGridOptions={"pagination": True},
    columnSize="sizeToFit",
)

appB.layout = html.Div([grid])


if __name__ == "__main__":
    makeup_scatter_plot.show()
    appA.run(debug=True)
    #appB.run(debug=True)