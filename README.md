# handson-dash
Handson sur plotly.Dash préparé pour le bdaday !

# 1. Setup your env

	./init_project_env.sh
	source venv/bin/activate

# 2. Enter the app zone

 - Open [app.py](./app.py)
 - import stuffs:


    import dash
    import dash_core_components as dcc
    import dash_html_components as html  

 - create the Dash's application (Flask like object)


    app = dash.Dash(__name__)
    
 - start the server*
 
 
    if __name__ == '__main__':
        app.run_server(debug=True)
        

*And run the app.py file*

 - add layout right after app creation (before starting the server)
 
 
    app.layout = html.Div([
        html.H1("Titanic Web App")
    ])
    
    
**PS : the code to start your server is always the last thing to do*

You can change your layout Python code dynamicaly ;)

# 3. Analytical Web App


We use the [titanic Dataset](http://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv) 
for the Hands on

First, append a [datatable](https://dash.plot.ly/datatabl) object
to your layout with the Titanic dataset loaded in it.

    app.layout = html.Div([
        html.H1("Titanic Web App"),
        # add here the code, DataTable(...)
    ])
    
With paging is User Friendly
    
    DataTable(
        ...
        pagination_settings={
            'current_page': 0,
            'page_size': 10
        }
    )

Scatter Box the Age column

    
# 4. SVM Explorer

