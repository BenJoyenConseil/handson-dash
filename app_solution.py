import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

# Add Boostrap CSS framework
app = dash.Dash(external_stylesheets=["https://stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css"])


def gen_charts(size, noise):
    """
    Compute plotly Graph Objects (train_scatter, auc_curve, confusion, contours, test_scatter) for a generated dataset
    with :size and :noise paraméters and a Moon pattern.
    :param size: int
    :param noise: int
    :return: train_scatter, auc_curve, confusion, contours, test_scatter
    """
    ### create dataset
    from sklearn import datasets
    X, y = datasets.make_moons(n_samples=size, noise=noise, random_state=0)

    ### plot initial X and y
    colors = [[0, 'red'], [1, 'blue']]
    train_scatter = go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(color=y, size=10, colorscale=colors),
        name='learned data'
    )

    ### train / test split
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

    ### model fit
    from sklearn.svm import SVC
    clf = SVC(kernel='rbf', gamma=2, C=1)
    clf.fit(X_train, y_train)

    ### plot metrics
    ##### ROC AUC
    from sklearn import metrics
    decision_test = clf.decision_function(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, decision_test)

    # AUC Score
    auc_score = metrics.roc_auc_score(y_true=y_test, y_score=decision_test)

    auc_curve = go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'Test Data {auc_score}',
    )
    y_pred_test = clf.predict(X_test)

    ##### Confusion Matrix
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

    ##### Contours
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
    test_scatter = go.Scatter(
        x=X_test[:, 0],
        y=X_test[:, 1],
        mode='markers',
        marker=dict(color=y_test, size=15, colorscale=colors, symbol='triangle-up'),
        name='test data'
    )

    return train_scatter, auc_curve, confusion, contours, test_scatter


# Control Panel
panel = html.Section(className='card', style=(dict(padding=20)), children=[
    html.P('Dataset size'),
    dcc.Slider(id='dataset-size', min=100, max=1000, value=500, step=100),
    html.P('Noise'),
    dcc.Slider(id='noise', min=0.0, max=1, value=0.3, step=0.1),
])

DEFAULT_LAYOUT = go.Layout(legend=dict(orientation='h'), margin=dict(l=10, r=10, t=5, b=50))

# Application Layout using Boostrap CSS grid system
app.layout = html.Div(className="container-fluid", children=[
    html.Nav(className='navbar', children=[
        html.H1("SVM Explorer"),
    ]),
    html.Div(className='row', children=[
        html.Div(id='graphs', className="col-10"),
        html.Div(className='col-2', children=panel)
    ])
])


# Called when the value of a slider (with id='dataset-size' or id='noise') changes
@app.callback(Output('graphs', 'children'),
              [Input('dataset-size', 'value'),
               Input('noise', 'value')])
def update_dataset_size(size, noise):
    train_scatter, auc_curve, confusion, contours, test_scatter, = gen_charts(size, noise)

    return [
        html.Div(className='row', children=[
            html.Div(className="col-4", children=[
                dcc.Graph(figure=go.Figure(data=[auc_curve], layout=DEFAULT_LAYOUT), style={'height': '30%'}),
                dcc.Graph(figure=go.Figure(data=[confusion], layout=DEFAULT_LAYOUT)),
            ]),
            html.Div(className="col-8", children=[
                dcc.Graph(className='h-100',
                          figure=go.Figure(data=[test_scatter, train_scatter, contours], layout=DEFAULT_LAYOUT)),
            ])
        ])
    ]


# Start the Flask server and listen on 8050
if __name__ == '__main__':
    app.run_server(debug=True)
