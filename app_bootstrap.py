from dash import Dash, html, dcc, dash_table, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
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

columnDefs = [
    { 'field': 'age'},
    { 'field': 'sex'},
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
    columnSize="autoSize",
    #skipHeader=True,
    defaultColDef={"resizable": True, "sortable": True, "filter": True, "skipHeader":True},
)

app.layout = dbc.Container([
    dbc.Container([
        #html.Br(),
        #html.Header("Cardiovascular Disease Prediction", style={'font-size': '24px'}),
        #html.Hr(),
        dbc.Row([ 
            html.Img(src=app.get_asset_url('intro_image.jpeg')),
            html.Br(),
            html.Br(),
            dbc.Col([
                html.Br(),
                #html.P("This dummy template demonstrates a monitoring UI for machine learning."),
                html.P("Medical data column descriptions:", style={'margin-left' : '15px'}),
                html.Li("cp: Chest pain type", id='cp', style={'margin-left' : '15px'}),
                html.Li("trestbps: Resting blood pressure", id='trestbps', style={'margin-left' : '15px'}),
                html.Li("chol: Serum cholesterol in mg/dl", id='chol', style={'margin-left' : '15px'}),
                html.Li("fbs: Fasting blood sugar > 120 mg/d", id='fbs', style={'margin-left' : '15px'}),
                html.Li("restecg: Resting electrocardiographic results", id='restecg', style={'margin-left' : '15px'}),
                html.Li("thalach: Maximum heart rate achieved", id='thalach', style={'margin-left' : '15px'}),
                html.Li("exang: Exercise induced angina", id='exang', style={'margin-left' : '15px'}),
                html.Li("oldpeak: ST depression induced by exercise relative to rest", id='oldpeak', style={'margin-left' : '15px'}),
                html.Li("slope: Of the peak exercise ST segment", id='slop', style={'margin-left' : '15px'}),
                html.Li("ca: Number of major vessels (0-3) colored by fluoroscopy", id='ca', style={'margin-left' : '15px'}),
                html.Li("thal: [normal; fixed defect; reversible defect]", id='thal', style={'margin-left' : '15px'}),
                html.Li("target: Heart disease yes / no", id='target', style={'margin-left' : '15px'}),
                html.Br(),
                html.Br(),
                html.Br(),
                dbc.Row([ 
                    dbc.Col([
                        html.Div(
                            [dash_table.DataTable(id='tbl_out', fill_width=False, style_header={
                                'backgroundColor': 'white',
                                'fontWeight': 'bold'
                            }, style_cell={'padding': '5px'},)],
                            style={'width': '50%', 'margin-left' : '15px'}
                        ),
                    ]),
                ], align='end'),
            ], style={'width': '45%'}),
            dbc.Col([
                dcc.Graph(figure=fig_corr, id='feature-corr', style={'width': '60vh', 'height': '48vh'})
            ]),
        ], justify="center"),
    ]),
    dbc.Row([ 
        html.Div([grid]), 
    ]),
    html.Br(),
    dbc.Container([
        dbc.Col([
                dbc.RadioItems(
                    options=[{'label': col, 'value': col} for col in df_cardiovascular.columns],
                    value='age',
                    id='feature-dist',
                    inline=True
                )
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure={}, id='feature-hist', style={'width': '115vh', 'height': '48vh'})
            ]),
        ], justify="center"),  # Use 'justify' to control spacing between columns
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(['Decision Tree', 'Random Forest', 'K-Nearest-Neighbor', 'Support Vector Machine'], ['Decision Tree', 'Random Forest'], id="model-dropdown", multi=True)
            ),
    ]),
    html.Br(),
    dbc.Row([
        html.Div(id='graph-model-results', children=[]),
    ]),
])

# Generate min/max/mean values for clicked column
@callback(
    Output(component_id='tbl_out', component_property='data'),
    Input(component_id='data_table', component_property='cellClicked')
)
def update_info(cellClicked):
    dict_meta_info = {'Column': 'none', 'Minimum': 'none', 'Maximum': 'none', 'Mean': 'none'}
    if cellClicked:
        # generate info on min/max in data column
        maximum = round(df_cardiovascular[cellClicked.get('colId')].max(), 4)
        minimum = round(df_cardiovascular[cellClicked.get('colId')].min(), 4)
        mean_col = round(df_cardiovascular[cellClicked.get('colId')].mean(), 4)
        dict_meta_info['Column'] = cellClicked.get('colId')
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