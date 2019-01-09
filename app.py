import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

app = dash.Dash(external_stylesheets=["https://stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css"])

from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=0)

colors = [[0, 'red'], [1, 'blue']]
data_points = go.Scatter(
    x=X[:, 0],
    y=X[:, 1],
    mode='markers',
    marker=dict(color=y, size=10, colorscale=colors),
    name='learned data'
)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
from sklearn.svm import SVC

clf = SVC(kernel='rbf', gamma=2, C=1)
clf.fit(X_train, y_train)
from sklearn import metrics

decision_test = clf.decision_function(X_test)
fpr, tpr, threshold = metrics.roc_curve(y_test, decision_test)

# AUC Score
auc_score = metrics.roc_auc_score(y_true=y_test, y_score=decision_test)

auc = go.Scatter(
    x=fpr,
    y=tpr,
    mode='lines',
    name='Test Diata',
)

y_pred_test = clf.predict(X_test)

matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_test)
tn, fp, fn, tp = matrix.ravel()

values = [tp, fn, fp, tn]
label_text = ["True Positive",
              "False Negative",
              "False Positive",
              "True Negative"]
labels = ["TP", "FN", "FP", "TN"]

colors = ['#66c2ff', '#0000cc', '#c65353', '#ff9999']

confusion = go.Pie(
    labels=label_text,
    values=values,
    hoverinfo='label+value+percent',
    textinfo='text+value',
    text=labels,
    sort=False,
    marker=dict(
        colors=colors
    )
)

import numpy as np

colors = [[0, 'red'], [1, 'blue']]
cm_bright = [[0, '#ff9999'], [1, '#66c2ff']]
h = .02
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

contours = go.Contour(
    x=np.arange(xx.min(), xx.max(), h),
    y=np.arange(yy.min(), yy.max(), h),
    z=Z.reshape(xx.shape),
    colorscale=cm_bright,
)
test_points = go.Scatter(
    x=X_test[:, 0],
    y=X_test[:, 1],
    mode='markers',
    marker=dict(color=y_test, size=15, colorscale=colors, symbol='triangle-up'),
    name='test data'
)

DEFAULT_LAYOUT = go.Layout(legend=dict(orientation='h'), margin=dict(l=0, r=0, t=0, b=0))

app.layout = html.Div([
    html.Div(className="container-fluid", children=[
        html.Nav(className='navbar', children=[
            html.H1("SVM Explorer"),
        ]),
        html.Div(className="row", children=[
            html.Div(className="col-3", children=[
                # dcc.Graph(id='train-data', className='h-25', figure=go.Figure(data=[data_points], layout=DEFAULT_LAYOUT)),
                dcc.Graph(id='auc', className='', figure=go.Figure(data=[auc], layout=DEFAULT_LAYOUT)),
                dcc.Graph(id='confusion-matrice', className='', figure=go.Figure(data=[confusion], layout=DEFAULT_LAYOUT)),
            ]),
            html.Div(className="col-7", children=[
                dcc.Graph(id='test-data', className='h-100',
                          figure=go.Figure(data=[contours, test_points, data_points], layout=DEFAULT_LAYOUT)),
            ]),
        ])
    ])
])
if __name__ == '__main__':
    app.run_server(debug=True)
