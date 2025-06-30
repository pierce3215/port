import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import datetime
import numpy as np

# --- Portfolio Data ---
portfolio = {
    'GOOGL': {'shares': 28, 'cost_basis': 185.45},
    'NVDA': {'shares': 100.012031, 'cost_basis': 113.46},
    'AMZN': {'shares': 5, 'cost_basis': 237},
    'INTC': {'shares': 3, 'cost_basis': 20},
    'CRM': {'shares': 29, 'cost_basis': 264.38}
}

# --- Functions ---
def get_portfolio_df():
    records = []
    total_value = 0
    total_cost = 0
    dividends = 0

    for ticker, pos in portfolio.items():
        stock = yf.Ticker(ticker)
        price = stock.fast_info.get('lastPrice')
        if price is None:
            continue

        shares = pos['shares']
        cost_basis = pos['cost_basis']
        value = shares * price
        invested = shares * cost_basis
        gain = value - invested

        # Simulated dividend income (replace with actual data if needed)
        dividend_rate = stock.info.get('dividendYield', 0) or 0
        annual_dividend = dividend_rate * value
        dividends += annual_dividend

        total_value += value
        total_cost += invested

        records.append({
            'Ticker': ticker,
            'Value': value,
            'Invested': invested,
            'Gain': gain,
            'Dividend': annual_dividend,
            'Sector': stock.info.get('sector', 'Unknown')
        })

    df = pd.DataFrame(records)
    df['Allocation'] = df['Value'] / total_value * 100
    return df, total_value, total_cost, dividends

def get_sector_breakdown(df):
    return df.groupby('Sector')['Value'].sum().reset_index()

def simulate_historical_growth(df, years=5, rate=0.12):
    now = datetime.now()
    dates = pd.date_range(end=now, periods=years, freq='Y')
    total = df['Value'].sum()
    values = [total / ((1 + rate) ** (years - i)) for i in range(years)]
    return pd.DataFrame({'Date': dates, 'Value': values})

# --- Dash App Setup ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Snowball Portfolio Tracker"

app.layout = dbc.Container([
    html.Br(),

    # --- Top Metrics ---
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Value", className="text-info"),
            dbc.CardBody([
                html.H3(id="total-value", className="card-title"),
                html.P(id="total-invested", className="card-text")
            ])
        ], color="dark", inverse=True), md=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Total Profit", className="text-success"),
            dbc.CardBody([
                html.H3(id="total-gain", className="card-title text-success"),
                html.P("", className="card-text")
            ])
        ], color="dark", inverse=True), md=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("IRR", className="text-primary"),
            dbc.CardBody([
                html.H3("8.51%", className="card-title text-primary"),
                html.P("5.93% current holdings", className="card-text")
            ])
        ], color="dark", inverse=True), md=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Passive Income", className="text-success"),
            dbc.CardBody([
                html.H3(id="total-dividend", className="card-title text-success"),
                html.P("Estimated annually", className="card-text")
            ])
        ], color="dark", inverse=True), md=3)
    ], className="mb-4"),

    # --- Charts ---
    dbc.Row([
        dbc.Col(dcc.Graph(id='donut-chart', config={'displayModeBar': False}), md=4),
        dbc.Col(html.Div(id='breakdown-table'))
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='sector-pie', config={'displayModeBar': False}), md=6),
        dbc.Col(dcc.Graph(id='growth-line', config={'displayModeBar': False}), md=6)
    ])

], fluid=True)

# --- Callbacks ---
@app.callback(
    [Output('total-value', 'children'),
     Output('total-invested', 'children'),
     Output('total-gain', 'children'),
     Output('total-dividend', 'children'),
     Output('donut-chart', 'figure'),
     Output('breakdown-table', 'children'),
     Output('sector-pie', 'figure'),
     Output('growth-line', 'figure')],
    Input('donut-chart', 'id')
)
def update_dashboard(_):
    df, total_value, total_cost, dividends = get_portfolio_df()
    gain = total_value - total_cost
    sector_df = get_sector_breakdown(df)
    growth_df = simulate_historical_growth(df)

    pie_fig = px.pie(df, names='Ticker', values='Value', hole=0.5)
    pie_fig.update_layout(template="plotly_dark")

    sector_fig = px.pie(sector_df, names='Sector', values='Value', hole=0.3, title='Sector Allocation')
    sector_fig.update_layout(template="plotly_dark")

    growth_fig = px.line(growth_df, x='Date', y='Value', title='Estimated Portfolio Growth')
    growth_fig.update_layout(template="plotly_dark")

    table_body = [html.Thead(html.Tr([
        html.Th("Name"),
        html.Th("Value/Invested"),
        html.Th("Gain"),
        html.Th("Allocation"),
        html.Th("Dividend")
    ]))]

    rows = []
    for _, row in df.iterrows():
        gain_color = "lime" if row['Gain'] >= 0 else "red"
        rows.append(html.Tr([
            html.Td(row['Ticker']),
            html.Td(f"${row['Value']:,.0f} / ${row['Invested']:,.0f}"),
            html.Td(f"{row['Gain']:,.2f}", style={"color": gain_color}),
            html.Td(f"{row['Allocation']:.2f}%"),
            html.Td(f"${row['Dividend']:,.2f}")
        ]))

    table_body.append(html.Tbody(rows))

    return (
        f"${total_value:,.2f}",
        f"${total_cost:,.2f} invested",
        f"{gain:+,.2f}",
        f"${dividends:,.2f}",
        pie_fig,
        dbc.Table(table_body, bordered=True, hover=True, responsive=True, striped=True, class_name="table-dark"),
        sector_fig,
        growth_fig
    )

if __name__ == '__main__':
    app.run(debug=True)