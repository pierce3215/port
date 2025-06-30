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

# Fixed reference date used across the dashboard
CURRENT_DATE = datetime(2025, 6, 30)

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
    total_value = 0.0
    total_cost = 0.0
    dividends = 0.0
    next_div_date = None
    next_div_amount = 0.0

    for ticker, pos in portfolio.items():
        stock = yf.Ticker(ticker)
        price = stock.fast_info.get("lastPrice")
        target = stock.info.get("targetMeanPrice")
        if price is None:
            continue

        shares = pos["shares"]
        cost_basis = pos["cost_basis"]
        value = shares * price
        invested = shares * cost_basis
        gain = value - invested

        # Dividend information
        dividend_rate = stock.info.get("dividendYield", 0) or 0
        annual_dividend = dividend_rate * value
        dividends += annual_dividend

        cal = stock.calendar or {}
        div_date = cal.get("Dividend Date")
        div_amount = 0.0
        if not stock.dividends.empty:
            div_amount = float(stock.dividends.iloc[-1])

        if div_date is not None:
            div_date = pd.to_datetime(div_date).date()
            if div_date >= CURRENT_DATE.date():
                total_payment = div_amount * shares
                if next_div_date is None or div_date < next_div_date:
                    next_div_date = div_date
                    next_div_amount = total_payment

        total_value += value
        total_cost += invested

        records.append(
            {
                "Ticker": ticker,
                "Shares": shares,
                "Cost Basis": cost_basis,
                "Value": value,
                "Invested": invested,
                "Gain": gain,
                "Dividend Income": annual_dividend,
                "Dividend Amount": div_amount,
                "Next Dividend Date": div_date,
                "Sector": stock.info.get("sector", "Unknown"),
                "Price Target": target,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df["Allocation"] = df["Value"] / total_value * 100
    return df, total_value, total_cost, dividends, next_div_date, next_div_amount

def get_sector_breakdown(df):
    return df.groupby('Sector')['Value'].sum().reset_index()

def simulate_historical_growth(df, years=5, rate=0.12, end_date=CURRENT_DATE):
    dates = pd.date_range(end=end_date, periods=years, freq="YE")
    total = df['Value'].sum()
    # reverse order so first date is oldest
    values = [total / ((1 + rate) ** (years - i - 1)) for i in range(years)]
    return pd.DataFrame({'Date': dates, 'Value': values})


def simulate_future_growth(current_value, years=5, rate=0.1, start_date=CURRENT_DATE):
    dates = pd.date_range(start=start_date, periods=years, freq="YE")
    values = [current_value * ((1 + rate) ** i) for i in range(1, years + 1)]
    return pd.DataFrame({"Date": dates, "Value": values})

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

    # --- Projection Inputs ---
    dbc.Row([
        dbc.Col(dbc.InputGroup([
            dbc.InputGroupText("Growth Rate %"),
            dbc.Input(id='growth-rate', type='number', value=10)
        ]), md=2),
        dbc.Col(dbc.InputGroup([
            dbc.InputGroupText("Years Ahead"),
            dbc.Input(id='growth-years', type='number', value=5)
        ]), md=2)
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

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Projected Value", className="text-info"),
            dbc.CardBody([
                html.H3(id='future-value', className='card-title'),
                html.P(id='future-date', className='card-text')
            ])
        ], color='dark', inverse=True), md=3)
    ], className='mb-4'),

    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Next Dividend", className="text-info"),
                dbc.CardBody([
                    html.H3(id="next-div-amount", className="card-title"),
                    html.P(id="next-div-date", className="card-text"),
                ]),
            ], color="dark", inverse=True),
            md=3,
        )
    ], className='mb-4'),

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
    [Output("total-value", "children"),
     Output("total-invested", "children"),
     Output("total-gain", "children"),
     Output("total-roi", "children"),
     Output("total-dividend", "children"),
     Output("future-value", "children"),
     Output("future-date", "children"),
     Output("next-div-amount", "children"),
     Output("next-div-date", "children"),
     Output("donut-chart", "figure"),
     Output("breakdown-table", "children"),
     Output("sector-pie", "figure"),
     Output("growth-line", "figure"),
     Output("gain-bar", "figure"),
     Output("remove-dropdown", "options"),
     Output("ticker-input", "value"),
     Output("shares-input", "value"),
     Output("cost-input", "value")],
    [Input('donut-chart', 'id'),
     Input('add-stock-btn', 'n_clicks'),
     Input('remove-stock-btn', 'n_clicks'),
     Input('growth-rate', 'value'),
     Input('growth-years', 'value')],
    [State('ticker-input', 'value'),
     State('shares-input', 'value'),
     State('cost-input', 'value'),
     State('remove-dropdown', 'value')]
)
def update_dashboard(_, add_clicks, remove_clicks, rate, years, ticker, shares, cost_basis, rem):
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
    df, total_value, total_cost, dividends, next_div_date, next_div_amount = get_portfolio_df()
    gain = total_value - total_cost
    roi = (gain / total_cost * 100) if total_cost else 0
    sector_df = get_sector_breakdown(df)
    hist_df = simulate_historical_growth(df)
    future_df = simulate_future_growth(total_value, years or 1, (rate or 0) / 100)
    growth_df = pd.concat([
        hist_df.assign(Type="Historical"),
        future_df.assign(Type="Projected"),
    ])
    growth_df.sort_values("Date", inplace=True)

    pie_fig = px.pie(df, names='Ticker', values='Value', hole=0.5)
    pie_fig.update_layout(template="plotly_dark")

    sector_fig = px.pie(sector_df, names='Sector', values='Value', hole=0.3, title='Sector Allocation')
    sector_fig.update_layout(template="plotly_dark")

    growth_fig = px.line(growth_df, x='Date', y='Value', color='Type', title='Portfolio Growth Projection')
    growth_fig.update_layout(template="plotly_dark")

    gain_bar = px.bar(df, x='Ticker', y='Gain', title='Gain by Ticker')
    gain_bar.update_layout(template="plotly_dark")

    table_body = [
        html.Thead(
            html.Tr(
                [
                    html.Th("Ticker"),
                    html.Th("Shares"),
                    html.Th("Cost Basis"),
                    html.Th("Value"),
                    html.Th("Invested"),
                    html.Th("Gain"),
                    html.Th("Allocation"),
                    html.Th("Dividend/Yr"),
                    html.Th("Next Div"),
                    html.Th("Price Target"),
                ]
            )
        )
    ]

    rows = []
    for _, row in df.iterrows():
        gain_color = "lime" if row["Gain"] >= 0 else "red"
        next_div = (
            pd.to_datetime(row["Next Dividend Date"]).strftime("%Y-%m-%d")
            if pd.notna(row["Next Dividend Date"])
            else "N/A"
        )
        rows.append(
            html.Tr(
                [
                    html.Td(row["Ticker"]),
                    html.Td(f"{row['Shares']:,.2f}"),
                    html.Td(f"${row['Cost Basis']:,.2f}"),
                    html.Td(f"${row['Value']:,.2f}"),
                    html.Td(f"${row['Invested']:,.2f}"),
                    html.Td(f"${row['Gain']:,.2f}", style={"color": gain_color}),
                    html.Td(f"{row['Allocation']:.2f}%"),
                    html.Td(f"${row['Dividend Income']:,.2f}"),
                    html.Td(next_div),
                    html.Td(f"${row['Price Target'] or 0:,.2f}"),
                ]
            )
        )

    table_body.append(html.Tbody(rows))

    future_value = future_df['Value'].iloc[-1] if not future_df.empty else total_value
    future_date = future_df['Date'].iloc[-1].strftime('%Y-%m-%d') if not future_df.empty else CURRENT_DATE.strftime('%Y-%m-%d')

    options = [{"label": t, "value": t} for t in portfolio.keys()]

    next_amt_text = f"${next_div_amount:,.2f}" if next_div_amount else "N/A"
    next_date_text = next_div_date.strftime("%Y-%m-%d") if next_div_date else "N/A"

    return (
        f"${total_value:,.2f}",
        f"${total_cost:,.2f} invested",
        f"{gain:+,.2f}",
        f"{roi:.2f}%",
        f"${dividends:,.2f}",
        f"${future_value:,.2f}",
        f"as of {future_date}",
        next_amt_text,
        next_date_text,
        pie_fig,
        dbc.Table(table_body, bordered=True, hover=True, responsive=True, striped=True, class_name="table-dark"),
        sector_fig,
        growth_fig,
        gain_bar,
        options,
        ticker,
        shares,
        cost_basis,
    )

if __name__ == '__main__':
    app.run(debug=True)