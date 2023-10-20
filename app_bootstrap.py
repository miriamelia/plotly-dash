from dash import Dash, html, dcc, dash_table, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import webbrowser
from main import figures_model_results
import dash_ag_grid as dag

# App idea is to present a prototype of a monitoring UI for a ML model in the medical context
df_cardiovascular = pd.read_csv('https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv')

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Design App here and add content from figures, dataset and other relevant calculations to respective divs
# https://wasmdash.vercel.app/


# Show feature correlations for selected attribute
fig_corr = px.imshow(df_cardiovascular.corr(), x=df_cardiovascular.columns, y=df_cardiovascular.columns, color_continuous_scale=px.colors.sequential.Cividis_r)
fig_corr.update_xaxes(side="top")
fig_corr.update_layout(width=800, height=600)

columnDefs = [
    { 'field': 'age' },
    { 'field': 'sex' },
    { 'field': 'cp'},
    { 'field': 'trestbps'},
    { 'field': 'chol'},
    { 'field': 'fbs'},
    { 'field': 'restecg'},
    { 'field': 'thalach'},
    { 'field': 'exang'},
    { 'field': 'oldpeak'},
    { 'field': 'slope'},
    { 'field': 'ca'},
    { 'field': 'thal'},
    { 'field': 'target'},
]

grid = dag.AgGrid(
    id='data_table',
    rowData=df_cardiovascular.to_dict("records"),
    columnDefs=columnDefs,
)

app.layout = dbc.Container([  # Wrap the layout in dbc.Container
    dbc.Container([
        html.Header("Cardiovascular Disease Prediction", style={'font-size': '24px'}),
        html.Hr(),
        dbc.Row([  # Use dbc.Row for horizontal layout
            dbc.Col([  # Use dbc.Col for column layout
                dbc.Table.from_dataframe(df_cardiovascular, striped=True, bordered=True, hover=True, id='tbl_out'),
            ]),
            dbc.Col([
                dbc.RadioItems(
                    options=[{'label': col, 'value': col} for col in df_cardiovascular.columns],
                    value='age',
                    id='feature-dist',
                    inline=True
                )
            ]),
        ]),
    ]),
    html.Br(),
    html.Div([grid]),
    #dbc.Table.from_dataframe(df_cardiovascular, striped=True, bordered=True, hover=True, responsive=True, id='data_table', responsive=True), # page_size=6, 
    html.Br(),
    dbc.Container([
        dbc.Row([  # Use dbc.Row for horizontal layout
            dbc.Col([
                dcc.Graph(figure=fig_corr, id='feature-corr')
            ], width=4),
            dbc.Col([
                dcc.Graph(figure={}, id='feature-hist', style={'width': '115vh', 'height': '48vh'})
            ], width=8),
        ], justify="between"),  # Use 'justify' to control spacing between columns
    ]),
    dbc.DropdownMenu(
        label="Select Models",
        children=[
            dbc.DropdownMenuItem("Decision Tree", id="Decision Tree"),
            dbc.DropdownMenuItem("Random Forest", id="Random Forest"),
            dbc.DropdownMenuItem("K-Nearest-Neighbor", id="K-Nearest-Neighbor"),
            dbc.DropdownMenuItem("Support Vector Machine", id="Support Vector Machine"),
        ],
        # multi=True,
        id="model-dropdown",
    ),
    html.Br(),
    html.Div(id='graph-model-results', children=[]),
])

# Generate min/max/mean values for clicked column
@callback(
    Output(component_id='tbl_out', component_property='data'),
    Input(component_id='data_table', component_property='active_cell')
)
def update_info(active_cell):
    dict_meta_info = {'Column': 'none', 'Minimum': 'none', 'Maximum': 'none', 'Mean': 'none'}
    if active_cell:
        # generate info on min/max in data column
        maximum = round(df_cardiovascular[active_cell.get('column_id')].max(), 4)
        minimum = round(df_cardiovascular[active_cell.get('column_id')].min(), 4)
        mean_col = round(df_cardiovascular[active_cell.get('column_id')].mean(), 4)
        dict_meta_info['Column'] = active_cell.get('column_id')
        dict_meta_info['Minimum'] = minimum
        dict_meta_info['Maximum'] = maximum
        dict_meta_info['Mean'] = mean_col
        # f"Column {active_cell.get('column_id')}: \n Min:{minimum} \t Max:{maximum} \t Mean:{mean_col}"
        return [dict_meta_info]
    return [dict_meta_info]

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
        fig = figures_model_results.get(value[i])
        graph = dcc.Graph(id=str(i), figure=fig)
        graphs.append(graph)
    return graphs

if __name__ == "__main__":
    app.run(debug=True)