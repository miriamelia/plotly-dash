import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# see https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data 
# see https://github.com/kb22/Heart-Disease-Prediction/blob/master/Heart%20Disease%20Prediction.ipynb 
df_cardiovascular = pd.read_csv('https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv') 
df_cardiovascular.info()
df_cardiovascular.describe()

# Understanding the Data 

# feature correlation matrix
fig = px.imshow(df_cardiovascular.corr(), x=df_cardiovascular.columns, y=df_cardiovascular.columns, color_continuous_scale=px.colors.sequential.Cividis_r)
fig.update_xaxes(side="top")
fig.update_layout(width=800, height=600)
fig.show()
fig.write_html('./figures/feature_correlation_matrix.html')

# histograms for each feature
# Create a subplot grid
n_cols = 7
n_rows = 2

fig = sp.make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=False)
fig.print_grid()

row_num = 1  # The row number for each subplot
count = 0
for i, column in enumerate(df_cardiovascular.columns):
    histogram = go.Histogram(x=df_cardiovascular[column], name=column)
    count = count + 1
    col_num = count # The column number for each subplot
    fig.add_trace(histogram, row=row_num, col=col_num)
    if ((i+1) % 7) == 0:
        row_num = row_num + 1
        count = 0
fig.update_layout(title_text="Feature Historgrams")
fig.show()
fig.write_html('./figures/feature_histograms.html')

# For target class
fig = px.histogram(df_cardiovascular, x='target')
fig.show()
fig.write_html('./figures/label_histogram.html')

# Data Preprocessing

# add dummy values
dataset = pd.get_dummies(df_cardiovascular, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
# scale data set
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

# Machine Learning
# Metrics.score returns accuracy

# separate label from features
y = df_cardiovascular['target']
X = df_cardiovascular.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# K_neighbors classifier
# with 20 different neighbor settings
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))

print("The score for K Neighbors Classifier is {}% with {} nieghbors.".format(knn_scores[np.array(knn_scores).argmax()]*100, 8))

# display scores for different neighbor values
fig = px.line(x=[i for i in range(1, 21)], y=knn_scores, markers=True, line_shape="linear", labels={"x": "Number of Neighbors (K)", "y": "Scores"})
fig.update_traces(marker=dict(color='red'))

for i in range(1, 21):
    fig.add_annotation(text=str((i, knn_scores[i - 1])), x=i, y=knn_scores[i - 1], showarrow=False)
fig.update_xaxes(tickvals=df_cardiovascular)
fig.update_layout(title="K Neighbors Classifier scores for different K values")
fig.show()
fig.write_html('./figures/knn_scores.html')

# Support Vector Classifier
svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel = kernels[i])
    svc_classifier.fit(X_train, y_train)
    svc_scores.append(svc_classifier.score(X_test, y_test))

print("The score for Support Vector Classifier is {}% with {} kernel.".format(svc_scores[np.array(svc_scores).argmax()]*100, 'linear'))

# Create a color scale for the bars
colors = px.colors.qualitative.Set1
bar_trace = go.Bar(x=kernels, y=svc_scores, marker=dict(color=colors))
text_annotations = [dict(x=i, y=score, text=score, showarrow=False) for i, score in enumerate(svc_scores)]
layout = go.Layout(title='Support Vector Classifier scores for different kernels',
                   xaxis=dict(title='Kernels'),
                   yaxis=dict(title='Scores'))
fig = go.Figure(data=[bar_trace], layout=layout)

for annotation in text_annotations:
    fig.add_annotation(annotation)

fig.show()
fig.write_html('./figures/svm_scores.html')

# Decision Tree
dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(dt_classifier.score(X_test, y_test))

print("The score for Decision Tree Classifier is {}% with {} maximum features.".format(dt_scores[np.array(dt_scores).argmax()]*100, [2,4,18]))

line_trace = go.Scatter(x=list(range(1, len(X.columns) + 1)), y=dt_scores, mode='lines+markers', line=dict(color='green'))
text_annotations = [dict(x=i, y=score, text=f'({i}, {score:.2f})', showarrow=False) for i, score in enumerate(dt_scores, 1)]

layout = go.Layout(
    title='Decision Tree Classifier scores for different number of maximum features',
    xaxis=dict(title='Max features'),
    yaxis=dict(title='Scores'),
)
fig = go.Figure(data=[line_trace], layout=layout)

for annotation in text_annotations:
    fig.add_annotation(annotation)

fig.update_xaxes(tickvals=list(range(1, len(X.columns) + 1)))
fig.show()
fig.write_html('./figures/dt_scores.html')

# Random Forest

rf_scores = []
estimators = [10, 100, 200, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(rf_classifier.score(X_test, y_test))

print("The score for Random Forest Classifier is {}% with {} estimators.".format(rf_scores[np.array(rf_scores).argmax()]*100, [100, 500]))

# Create a color scale for the bars
colors = px.colors.qualitative.Set1
bar_trace = go.Bar(
    x=[str(estimator) for estimator in estimators],
    y=rf_scores,
    marker=dict(color=colors),
    width=0.8
)
text_annotations = [dict(x=i, y=score, text=score, showarrow=False) for i, score in enumerate(rf_scores)]
layout = go.Layout(
    title='Random Forest Classifier scores for different number of estimators',
    xaxis=dict(title='Number of estimators'),
    yaxis=dict(title='Scores')
)

fig = go.Figure(data=[bar_trace], layout=layout)
for annotation in text_annotations:
    fig.add_annotation(annotation)

fig.show()
fig.write_html('./figures/rf_scores.html')
