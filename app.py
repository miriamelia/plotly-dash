from dash import Dash, html, dcc, dash_table, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import webbrowser
import dash_dangerously_set_inner_html
from main import figures_model_results

# App idea is to present a prototype of a monitoring UI for a ML model in the medical context
df_cardiovascular = pd.read_csv('https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv')

app = Dash(__name__)

# Design App here and add content from figures, dataset and other relevant calculations to respective divs
# https://wasmdash.vercel.app/

app.layout = html.Div([
    html.Div(children='Cardiovascular Disease Prediction'),
    html.Hr(),
    dcc.RadioItems(options=df_cardiovascular.columns, value='age', id='feature-dist', inline=True),
    html.Br(),
    dbc.Alert(id='tbl_out', color='primary'),
    html.Br(),
    dash_table.DataTable(data=df_cardiovascular.to_dict('records'), page_size=6, id='data_table'),
    dcc.Graph(figure={}, id='feature-hist'),
    html.Br(), 
    dcc.Dropdown(['Decision Tree', 'Random Forest', 'K-Nearest-Neighbor', 'Support Vector Machine'], ['Decision Tree', 'Random Forest'], id="model-dropdown", multi=True), 
    html.Br(),
    html.Div(id='graph-model-results', children=[])
])

# Generate min/max/mean values for clicked column
@callback(
    Output(component_id='tbl_out', component_property='children'),
    Input(component_id='data_table', component_property='active_cell')
)
def update_info(active_cell):
    print(active_cell)
    if active_cell:
        # generate info on min/max in data column
        maximum = round(df_cardiovascular[active_cell.get('column_id')].max(), 4)
        minimum = round(df_cardiovascular[active_cell.get('column_id')].min(), 4)
        mean_col = round(df_cardiovascular[active_cell.get('column_id')].mean(), 4)
        return f"Column {active_cell.get('column_id')}: \n Min:{minimum} \t Max:{maximum} \t Mean:{mean_col}"
    return "Column: \n Min: \t Max: \t Avg:"

# Show feature historgam for selected attribute
@callback(
    Output(component_id='feature-hist', component_property='figure'),
    Input(component_id='feature-dist', component_property='value')
)
def update_graph(col_chosen):
    fig = px.histogram(df_cardiovascular, x=df_cardiovascular[col_chosen], histfunc='count')
    return fig

# Display sleected model scores
@callback(Output('graph-model-results', 'children'),
          Input('model-dropdown', 'value'))
def update_img_source(value):
    graphs = []
    for i in range(len(value)):
        print(i)
        fig = figures_model_results.get(value[i])
        graph = dcc.Graph(id=str(i), figure=fig)
        graphs.append(graph)
    return graphs

if __name__ == "__main__":
    app.run(debug=True)