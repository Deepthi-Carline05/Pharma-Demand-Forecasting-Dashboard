import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("salesmonthly.csv")

df['datum'] = pd.to_datetime(df['datum'])
df = df.sort_values('datum')
df = df.reset_index(drop=True)

# Drug category columns
sales_columns = ['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']

# Create Total Sales
df['Total_Sales'] = df[sales_columns].sum(axis=1)

# -----------------------------
# Initialize App
# -----------------------------
app = dash.Dash(__name__)

app.layout = html.Div([

    html.H1("Pharma Sales Intelligence Portal", 
            style={'textAlign':'center'}),

    html.Label("Select Category:"),

    dcc.Dropdown(
        id='category-dropdown',
        options=[{'label': col, 'value': col} for col in ['Total_Sales'] + sales_columns],
        value='Total_Sales'
    ),

    html.Br(),

    # KPI CARDS
    html.Div([
        html.Div([
            html.H4("Latest Sales"),
            html.H2(id="latest-sales")
        ], style={"width":"30%","display":"inline-block","textAlign":"center"}),

        html.Div([
            html.H4("Forecast Next Month"),
            html.H2(id="forecast-sales")
        ], style={"width":"30%","display":"inline-block","textAlign":"center"}),

        html.Div([
            html.H4("Growth %"),
            html.H2(id="growth-rate")
        ], style={"width":"30%","display":"inline-block","textAlign":"center"})
    ]),

    html.Br(),

    dcc.Graph(id='sales-graph'),

    html.Br(),

    html.Label("Forecast Growth Adjustment (%)"),

    dcc.Slider(
        id='growth-slider',
        min=0,
        max=30,
        step=1,
        value=0,
        marks={i: f"{i}%" for i in range(0,31,5)}
    )
])

# -----------------------------
# Callback
# -----------------------------
@app.callback(
    [Output('sales-graph','figure'),
     Output("latest-sales","children"),
     Output("forecast-sales","children"),
     Output("growth-rate","children")],
    [Input('category-dropdown','value'),
     Input('growth-slider','value')]
)
def update_graph(selected_category, growth_adjustment):

    filtered_df = df[['datum', selected_category]].copy()

    # Create lag features
    filtered_df['Lag_1'] = filtered_df[selected_category].shift(1)
    filtered_df['Lag_2'] = filtered_df[selected_category].shift(2)
    filtered_df = filtered_df.dropna()

    # Train-test split
    train_size = int(len(filtered_df) * 0.8)
    train = filtered_df.iloc[:train_size]
    test = filtered_df.iloc[train_size:]

    # Train model
    model = LinearRegression()
    model.fit(train[['Lag_1','Lag_2']], train[selected_category])

    predictions = model.predict(test[['Lag_1','Lag_2']])

    # Apply growth simulation
    predictions = predictions * (1 + growth_adjustment/100)

    # KPI calculations
    latest_value = filtered_df[selected_category].iloc[-1]
    forecast_value = predictions[-1]
    growth = ((forecast_value - latest_value) / latest_value) * 100

    # Graph
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_df['datum'],
        y=filtered_df[selected_category],
        mode='lines',
        name='Historical Sales'
    ))

    fig.add_trace(go.Scatter(
        x=test['datum'],
        y=predictions,
        mode='lines+markers',
        name='Forecast'
    ))

    fig.update_layout(
        template='plotly_white',
        xaxis_title="Date",
        yaxis_title="Sales"
    )

    return fig, f"{latest_value:.0f}", f"{forecast_value:.0f}", f"{growth:.2f}%"

# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)