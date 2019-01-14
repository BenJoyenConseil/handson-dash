Handson-dash
===
Hands on plotly.Dash, prepared for the OCTO's BDA day !

It is inspired from [the scikit-learn Classifier comparaison topic](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py)
and the wonderfull [demo of Dash : SVM Explorer](https://dash-gallery.plotly.host/dash-svm)

# 1. Setup your env

	./init_project_env.sh
	source venv/bin/activate

# 2. Enter the app zone

 - Open [app.py](./app.py)
 - import stuffs:


	import dash
	import dash_core_components as dcc # Core components (Graph..)
	import dash_html_components as html   # Html components for the layout (Div, H1)

 - create the Dash's application (Flask like object)


	app = dash.Dash(__name__)
    
 - start the server*
 
 
    if __name__ == '__main__':
        app.run_server(debug=True)
        

*And run the app.py file*

 - add layout right after app creation (before starting the server)
 
 
	app.layout = html.Div([
		html.H1("SVM Explorer")
	])
    
    
**PS : the code to start your server is always the last thing to do*

You can change your layout Python code dynamicaly ;)

# 3. Analytical Web App
    
## Title : SVM Explorer

We have an existing notebook nammed [bjc_svm_notebook](bjc_svm_exploration.ipynb).
It uses the plotly library to plot some charts about an SVM Classifier trained on faked data.

**Our goal** : to migrate an existing notebook datavisualization 
into an interactive web applictation

1. Open the notebook and run its cells

        jupyter notebook

2. Move existing charts into [app.py](app_solution.py)

    For example: to plot a Scatter trace with Dash, see [dcc.Graph](https://dash.plot.ly/dash-core-components)
    
3. Use [Boostrap CSS framework](https://getbootstrap.com/docs/4.2/layout/overview/) to organize charts using grid system   

*include Boostrap CSS into the frontend*

    app = dash.Dash(external_stylesheets=["https://stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css"])

*Layout*

    app.layout = html.Div(className="container-fluid", children=[
        html.Nav(className='navbar', children=[
            html.H1("SVM Explorer"),
        ]),
        html.Div(className='row', children=[
            # Graphs ...
        ])
    ])

4. Add interactivity with 2 [sliders](https://dash.plot.ly/dash-core-components/slider)
 that controls the parameters of the dataset generation.
 
*it should look like this*

	# Panel
	panel = html.Section( className='card', style=(dict(padding=20)), children=[
        html.P('Dataset size'),
        dcc.Slider(id='dataset-size',min=100, max=1000, value=500, step=100),
        html.P('Noise'),
        dcc.Slider(id='noise', min=0.0, max=1, value=0.3, step=0.1),
	])
	...
	# Layout
	app.layout = html.Div(className="container-fluid", children=[
        html.Nav(className='navbar', children=[
            html.H1("SVM Explorer"),
        ]),
        html.Div(className='row', children=[
            html.Div(id='graphs', className="col-10"),
            html.Div(className='col-2', children=panel)
        ])
    ])

5. Add controls for model's parameter (kernel, gamma, C)

6. Play using these [SVM parameters best practices](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)

7. Add a save button that pickles the model with parameters
