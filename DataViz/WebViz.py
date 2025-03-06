# Web Data Visualization Techniques for Data Analysts

# 1. Importing Essential Libraries
import pandas as pd
import numpy as np
from flask import Flask, render_template_string
import plotly.express as px
import plotly.graph_objects as go
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components

# Sample Data for Visualization
# Creating a simple DataFrame for demonstration purposes
data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D', 'E'],
    'Values': [10, 23, 45, 32, 17],
    'Groups': ['G1', 'G2', 'G1', 'G2', 'G1'],
    'Time': pd.date_range(start='2023-01-01', periods=5, freq='D')
})
print("Sample Data:\n", data)

# 2. Flask Setup for Web Visualization
app = Flask(__name__)

# 3. Plotly Visualization
# Example of an interactive bar chart using Plotly
def plotly_bar_chart(data):
    fig = px.bar(data, x='Category', y='Values', color='Groups', title='Interactive Bar Chart with Plotly')
    return fig.to_html(full_html=False)

# 4. Bokeh Visualization
# Example of an interactive line chart using Bokeh
def bokeh_line_chart(data):
    p = figure(title="Bokeh Line Chart", x_axis_type="datetime", plot_width=800, plot_height=400)
    p.line(data['Time'], data['Values'], line_width=2, line_color='green')
    script, div = components(p)
    return script, div

# 5. Flask Route for Web Page
@app.route('/')
def index():
    plotly_chart = plotly_bar_chart(data)
    bokeh_script, bokeh_div = bokeh_line_chart(data)
    html_template = f'''
    <html>
        <head>
            <title>Web Data Visualization</title>
        </head>
        <body>
            <h1>Web Data Visualization Examples</h1>
            <h2>Plotly Chart</h2>
            {plotly_chart}
            <h2>Bokeh Chart</h2>
            {bokeh_script}
            {bokeh_div}
        </body>
    </html>
    '''
    return render_template_string(html_template)

# 6. Running the Flask Application
if __name__ == '__main__':
    app.run(debug=True)