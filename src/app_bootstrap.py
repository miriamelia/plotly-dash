from dash import Dash, html, dcc, dash_table, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from main import figures_model_results
import dash_ag_grid as dag

# App idea is to present a prototype of a monitoring UI for a ML model in the medical context
# Implemented as dash bootstrap components, find app here https://heart-disease-prediction-ml-dashboard.onrender.com/ 
# Based on:
# see https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data 
# see https://github.com/kb22/Heart-Disease-Prediction/blob/master/Heart%20Disease%20Prediction.ipynb 

df_cardiovascular = pd.read_csv('https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv')

# add bootstrap theme
app = Dash("heart_disease_prediction", external_stylesheets=[dbc.themes.BOOTSTRAP])
# for running onrender server
server = app.server

# Understanding the Data 

# Show feature correlations for selected attribute
fig_corr = px.imshow(df_cardiovascular.corr(), x=df_cardiovascular.columns, y=df_cardiovascular.columns, color_continuous_scale=px.colors.sequential.Cividis_r)
fig_corr.update_xaxes(side="top")

# dataset
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

# display data set
grid = dag.AgGrid(
    id='data_table',
    rowData=df_cardiovascular.to_dict("records"),
    columnDefs=columnDefs,
    columnSize="autoSize",
    defaultColDef={"resizable": True, "sortable": True, "filter": True, "skipHeader":True},
)

# display medical column definitions as list
features = {'cp': 'cp: Chest pain type', 'trestbps': 'trestbps: Resting blood pressure', 'chol': 'chol: Serum cholesterol in mg/dl', 'fbs':'fbs: Fasting blood sugar > 120 mg/d', 'restecg': 'restecg: Resting electrocardiographic results', 'thalach':'thalach: Maximum heart rate achieved', 'exang':'exang: Exercise induced angina', 'oldpeak':'oldpeak: ST depression induced by exercise relative to rest', 'slope':'slope: Of the peak exercise ST segment', 'ca':'ca: Number of major vessels (0-3) colored by fluoroscopy', 'thal':'thal: [normal; fixed defect; reversible defect]', 'target':'target: Heart disease yes / no'}
list_items = []
for key, val in features.items():
    new_val = dbc.ListGroupItem(str(val), id=str(key))
    list_items.append(new_val)

# define layout tab data - 5 rows with 1 or 2 cols
tab_data = dbc.Card(
    dbc.CardBody([
        dbc.Container([
            # Row 1 - 2 cols
            dbc.Row([ 
                # display intro image
                html.Img(src=app.get_asset_url('intro_image.jpeg')),
                html.Br(),
                html.Br(),
                dbc.Col([
                    html.Br(),
                    html.H2(["Medical column definitions", 
                        # Link to original data set
                        dbc.Badge(
                            "Data", 
                            href='https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data',
                            className="ms-1",
                        )], style={'text-align':'center'}),
                    # display medical column definitions
                    dbc.ListGroup(children=list_items, id="selectedFeature", style={'margin-left' : '15px'}),
                ]),
                # display feature correlation matrix
                dbc.Col([
                    html.Br(),
                    html.H2("Feature Correlation",  style={'text-align': 'center'}),
                    dcc.Graph(figure=fig_corr, id='feature-corr', style={'width': '60vh', 'height': '52vh'}),
                ]),
            ], justify="evenly"),
        ]),
        html.Br(),
        # Row 2 - 1 col
        dbc.Row([ 
            # display data table
            html.Div([grid]), 
        ]),
        html.Br(),
        # Row 3 - 2 cols
        dbc.Row([
            dbc.Col([
                # display select col-name
                dbc.RadioItems(
                    options=[{'label': col, 'value': col} for col in df_cardiovascular.columns],
                    value='age',
                    id='feature-dist',
                    inline=True
                ), 
            ], width=8),
            # display min/max/mean information on column value distribution in table
            dbc.Col([ 
                html.Div(
                    [dash_table.DataTable(id='tbl_out', fill_width=False, cell_selectable=False, style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold'
                    }, style_cell={'padding': '5px', 'textAlign': 'right'},)],
                ),
            ], width=4),
        ], justify="between"),
        # Row 4 - 1 col
        dbc.Row([
            dbc.Col([
                # display feature histogram
                dcc.Graph(figure={}, id='feature-hist', style={'width': '110vh', 'height': '48vh'})
            ]),
        ], justify="evenly"),
        # Row 5 - 1 col
        dbc.Row([
            dbc.Col(
                # footer
                html.Footer("© Miriam Elia & Regina Kushtanova", style={'text-align': 'center'}),
        )], align='evenly'),
    ]),
    className="mt-3",
)

# define layout tab model - 5 rows with 1 or 2 cols
tab_model = dbc.Card(
    dbc.CardBody([
        # Row 1 - 1 col
        dbc.Row([
            # display intro image
            html.Img(src=app.get_asset_url('intro_image.jpeg')),
        ]),
        html.Br(),
        # Row 2 - 1 col
        dbc.Row([
            html.H2(["Model Selection", 
                # display Title and Link to code 
                dbc.Badge(
                    "Code", 
                    href='https://github.com/kb22/Heart-Disease-Prediction/blob/master/Heart%20Disease%20Prediction.ipynb',
                    className="ms-1",
                )], style={'text-align':'center'}),
        ]),
        html.Br(),
        # Row 2 - 1 col
        dbc.Row([
            # display dropdown 
            dbc.Col(dcc.Dropdown(['Decision Tree', 'Random Forest', 'K-Nearest-Neighbor', 'Support Vector Machine'], ['Decision Tree', 'Random Forest'], id="model-dropdown", multi=True)),
        ]),
        html.Br(),
        # Row 3 - 1 col 
        dbc.Row([
            # display graphs
            html.Div(id='graph-model-results', children=[]),
        ]),
        # Row 4 - 1 col
       dbc.Row([
            dbc.Col(
                # footer
                html.Footer("© Miriam Elia & Regina Kushtanova", style={'text-align': 'center'}),
        )], align='evenly'),
    ]),
    className="mt-3",
)

# bring tabs together
app.layout = dbc.Container([
    dbc.Tabs(
    [
        dbc.Tab(tab_data, label="Data"),
        dbc.Tab(tab_model, label="Model"),
    ])
])

# Data Tab

# Generate min/max/mean values for clicked column
@callback(
    Output(component_id='tbl_out', component_property='data'),
    Input(component_id='data_table', component_property='cellClicked')
)
# cellClicked is dict: {'value': 1, 'colId': 'slope', 'rowIndex': 5, 'rowId': '5', 'timestamp': 1698313876764}
def update_info(cellClicked):
    dict_meta_info = {'Column': '/', 'Minimum': '0', 'Maximum': '0', 'Mean': '0'}
    if cellClicked:
        # generate info on min/max/mean in data column via ag_grid component property 'cellClicked'
        maximum = round(df_cardiovascular[cellClicked.get('colId')].max(), 4)
        minimum = round(df_cardiovascular[cellClicked.get('colId')].min(), 4)
        mean_col = round(df_cardiovascular[cellClicked.get('colId')].mean(), 4)
        dict_meta_info['Column'] = cellClicked.get('colId')
        dict_meta_info['Minimum'] = minimum
        dict_meta_info['Maximum'] = maximum
        dict_meta_info['Mean'] = mean_col
        # display generated data in component property 'data' of data_table
        return [dict_meta_info]
    return [dict_meta_info]

# Highlist selected list item
@callback(
    Output(component_id='selectedFeature', component_property='children'),
    Input(component_id='data_table', component_property='cellClicked'),
)
# cellClicked is dict: {'value': 1, 'colId': 'slope', 'rowIndex': 5, 'rowId': '5', 'timestamp': 1698313876764}
def update_info(cellClicked):
    # generate text for list items to be displayed
    feature_list = {'cp': 'cp: Chest pain type', 'trestbps': 'trestbps: Resting blood pressure', 'chol': 'chol: Serum cholesterol in mg/dl', 'fbs':'fbs: Fasting blood sugar > 120 mg/d', 'restecg': 'restecg: Resting electrocardiographic results', 'thalach':'thalach: Maximum heart rate achieved', 'exang':'exang: Exercise induced angina', 'oldpeak':'oldpeak: ST depression induced by exercise relative to rest', 'slope':'slope: Of the peak exercise ST segment', 'ca':'ca: Number of major vessels (0-3) colored by fluoroscopy', 'thal':'thal: [normal; fixed defect; reversible defect]', 'target':'target: Heart disease yes / no'}
    if cellClicked != None:
        # get columns name of clicked cell in ag grid
        feature = cellClicked.get('colId')
        active_list = []
        if feature == 'age' or feature == 'sex':
            return list_items
        # Create dbc.ListGroupItems based on feature_list text
        for key, val in feature_list.items():
            new_val = dbc.ListGroupItem(str(val), id=str(key))
            active_list.append(new_val)
        if cellClicked != None:
            # modify layout of selected column name to highlight color
            active_list[list(feature_list).index(feature)] = dbc.ListGroupItem(str(feature_list.get(feature)), id=str(feature), color='warning')
            # return list of ListGroupItems as component property children to ListGroup in layout 
            return active_list
    return list_items

# Show feature historgam for selected attribute
@callback(
    Output(component_id='feature-hist', component_property='figure'),
    Input(component_id='feature-dist', component_property='value')
)
def update_graph(col_chosen):
    fig = px.histogram(df_cardiovascular, x=df_cardiovascular[col_chosen], histfunc='count')
    return fig

# Model Tab

# Display sleected model scores
@callback(Output('graph-model-results', 'children'),
          Input('model-dropdown', 'value'))
# value is list of selectable values in dropdown - here multi-select
def update_img_source(value):
    graphs = []
    # generate graphs for selected dropdownitems 
    for i in range(len(value)):
        # figures_model_results is dict with key dropdownitem-name and value generated graph imported from main.py
        fig = figures_model_results.get(value[i])
        graph = dcc.Graph(id=str(i), figure=fig)
        graphs.append(graph)
    # return list of graphs as component property children of html.div 
    return graphs

if __name__ == "__main__":
    app.run(debug=True)