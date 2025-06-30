import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import datetime
import numpy as np
import json
import os

# --- Portfolio Data ---
PORTFOLIO_FILE = "portfolio.json"

DEFAULT_PORTFOLIO = {
    "GOOGL": {"shares": 28, "cost_basis": 185.45},
    "NVDA": {"shares": 100.012031, "cost_basis": 113.46},
    "AMZN": {"shares": 5, "cost_basis": 237},
    "INTC": {"shares": 3, "cost_basis": 20},
    "CRM": {"shares": 29, "cost_basis": 264.38},
}


def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_PORTFOLIO.copy()


def save_portfolio(data: dict) -> None:
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2)


portfolio = load_portfolio()

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

    # --- Add Stock ---
    dbc.Row([
        dbc.Col(dbc.InputGroup([
            dbc.InputGroupText("Ticker"),
            dbc.Input(id="ticker-input", placeholder="AAPL", type="text")
        ]), md=2),
        dbc.Col(dbc.InputGroup([
            dbc.InputGroupText("Shares"),
            dbc.Input(id="shares-input", placeholder="0", type="number")
        ]), md=2),
        dbc.Col(dbc.InputGroup([
            dbc.InputGroupText("Cost"),
            dbc.Input(id="cost-input", placeholder="0", type="number")
        ]), md=2),
        dbc.Col(dbc.Button('Add Stock', id='add-stock-btn', color='success', className='w-100'), md=2)
    ], className='mb-2'),

    # --- Remove Stock ---
    dbc.Row([
        dbc.Col(dcc.Dropdown(id='remove-dropdown', placeholder='Select to remove', options=[{"label": t, "value": t} for t in portfolio.keys()]), md=3),
        dbc.Col(dbc.Button('Remove Stock', id='remove-stock-btn', color='danger', className='w-100'), md=2)
    ], className='mb-4'),
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
            dbc.CardHeader("ROI", className="text-primary"),
            dbc.CardBody([
                html.H3(id="total-roi", className="card-title text-primary"),
                html.P("Return on investment", className="card-text")
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
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='gain-bar', config={'displayModeBar': False}), md=12)
    ], className='mb-4')

], fluid=True)

# --- Callbacks ---
@app.callback(
    [Output('total-value', 'children'),
     Output('total-invested', 'children'),
     Output('total-gain', 'children'),
     Output('total-roi', 'children'),
     Output('total-dividend', 'children'),
     Output('donut-chart', 'figure'),
     Output('breakdown-table', 'children'),
     Output('sector-pie', 'figure'),
     Output('growth-line', 'figure'),
     Output('gain-bar', 'figure'),
     Output('remove-dropdown', 'options'),
     Output('ticker-input', 'value'),
     Output('shares-input', 'value'),
     Output('cost-input', 'value')],
    [Input('donut-chart', 'id'),
     Input('add-stock-btn', 'n_clicks'),
     Input('remove-stock-btn', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('shares-input', 'value'),
     State('cost-input', 'value'),
     State('remove-dropdown', 'value')]
)
def update_dashboard(_, add_clicks, remove_clicks, ticker, shares, cost_basis, rem):
    ctx = callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    if trigger == 'add-stock-btn' and ticker and shares and cost_basis:
        portfolio[ticker.upper()] = {
            'shares': float(shares),
            'cost_basis': float(cost_basis)
        }
        save_portfolio(portfolio)
        ticker = shares = cost_basis = ''
    elif trigger == 'remove-stock-btn' and rem:
        portfolio.pop(rem, None)
        save_portfolio(portfolio)
    df, total_value, total_cost, dividends = get_portfolio_df()
    gain = total_value - total_cost
    roi = (gain / total_cost * 100) if total_cost else 0
    sector_df = get_sector_breakdown(df)
    growth_df = simulate_historical_growth(df)

    pie_fig = px.pie(df, names='Ticker', values='Value', hole=0.5)
    pie_fig.update_layout(template="plotly_dark")

    sector_fig = px.pie(sector_df, names='Sector', values='Value', hole=0.3, title='Sector Allocation')
    sector_fig.update_layout(template="plotly_dark")

    growth_fig = px.line(growth_df, x='Date', y='Value', title='Estimated Portfolio Growth')
    growth_fig.update_layout(template="plotly_dark")

    gain_bar = px.bar(df, x='Ticker', y='Gain', title='Gain by Ticker')
    gain_bar.update_layout(template="plotly_dark")

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

    options = [{"label": t, "value": t} for t in portfolio.keys()]

    return (
        f"${total_value:,.2f}",
        f"${total_cost:,.2f} invested",
        f"{gain:+,.2f}",
        f"{roi:.2f}%",
        f"${dividends:,.2f}",
        pie_fig,
        dbc.Table(table_body, bordered=True, hover=True, responsive=True, striped=True, class_name="table-dark"),
        sector_fig,
        growth_fig,
        gain_bar,
        options,
        ticker,
        shares,
        cost_basis
    )

if __name__ == '__main__':
    app.run(debug=True)