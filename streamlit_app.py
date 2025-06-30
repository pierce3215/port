import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from datetime import datetime
import json
import os

# --- Portfolio Data ---
PORTFOLIO_FILE = "portfolio.json"
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


@st.cache_resource(ttl=3600)
def fetch_stock(ticker: str) -> yf.Ticker:
    """Fetch a ``yfinance.Ticker`` object and cache it for an hour.

    ``st.cache_data`` attempts to pickle return values which fails with the
    ``yfinance.Ticker`` object. ``st.cache_resource`` avoids pickling and is
    therefore safe to use here.
    """
    return yf.Ticker(ticker)


def get_portfolio_df(portfolio):
    records = []
    total_value = 0.0
    total_cost = 0.0
    dividends = 0.0
    next_div_date = None
    next_div_amount = 0.0

    for ticker, pos in portfolio.items():
        stock = fetch_stock(ticker)
        price = stock.fast_info.get("lastPrice")
        info = stock.info or {}
        target = info.get("targetMeanPrice")
        if price is None:
            continue

        shares = pos["shares"]
        cost_basis = pos["cost_basis"]
        value = shares * price
        invested = shares * cost_basis
        gain = value - invested

        div_rate = info.get("dividendRate") or 0
        div_yield = (info.get("dividendYield") or 0) * 100
        annual_dividend = div_rate * shares
        dividends += annual_dividend

        hist = stock.history(period="1y")
        change_1y = 0.0
        if not hist.empty:
            first_price = hist["Close"].iloc[0]
            if first_price:
                change_1y = (price - first_price) / first_price * 100

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
                "Price": price,
                "Value": value,
                "Invested": invested,
                "Gain": gain,
                "Dividend Yield %": div_yield,
                "Dividend Income": annual_dividend,
                "Dividend Amount": div_amount,
                "Next Dividend Date": div_date,
                "Sector": info.get("sector", "Unknown"),
                "Price Target": target,
                "PE Ratio": info.get("trailingPE") or info.get("forwardPE"),
                "52w Low": info.get("fiftyTwoWeekLow"),
                "52w High": info.get("fiftyTwoWeekHigh"),
                "Beta": info.get("beta"),
                "1Y Change %": change_1y,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df["Allocation"] = df["Value"] / total_value * 100
    return df, total_value, total_cost, dividends, next_div_date, next_div_amount


def get_sector_breakdown(df):
    return df.groupby("Sector")["Value"].sum().reset_index()


def simulate_historical_growth(df, years=5, rate=0.12, end_date=CURRENT_DATE):
    """Generate a simple historical growth series.

    Using ``DateOffset`` ensures the final point is ``end_date`` rather than the
    previous year end. This avoids a gap in the chart when ``CURRENT_DATE`` is
    mid-year.
    """
    freq = pd.DateOffset(years=1)
    dates = pd.date_range(end=end_date, periods=years, freq=freq)
    total = df["Value"].sum()
    values = [total / ((1 + rate) ** (years - i - 1)) for i in range(years)]
    return pd.DataFrame({"Date": dates, "Value": values})


def simulate_future_growth(
    current_value, years=5, rate=0.1, start_date=CURRENT_DATE
):
    """Project future growth starting one year after ``start_date``."""
    freq = pd.DateOffset(years=1)
    dates = pd.date_range(start=start_date + freq, periods=years, freq=freq)
    values = [current_value * ((1 + rate) ** i) for i in range(1, years + 1)]
    return pd.DataFrame({"Date": dates, "Value": values})


def main():
    st.set_page_config(page_title="Snowball Portfolio Tracker", layout="wide")
    st.title("Snowball Portfolio Tracker")

    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = load_portfolio()
    portfolio = st.session_state["portfolio"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Add Stock")
        with st.form("add_stock", clear_on_submit=True):
            ticker = st.text_input("Ticker")
            shares = st.number_input("Shares", min_value=0.0, step=1.0)
            cost = st.number_input("Cost Basis", min_value=0.0, step=0.01)
            submitted = st.form_submit_button("Add")
            if submitted and ticker and shares and cost:
                portfolio[ticker.upper()] = {"shares": float(shares), "cost_basis": float(cost)}
                save_portfolio(portfolio)
                st.experimental_rerun()

    with col2:
        st.subheader("Remove Stock")
        with st.form("remove_stock"):
            options = list(portfolio.keys())
            to_remove = st.selectbox("Select ticker", options) if options else None
            remove_btn = st.form_submit_button("Remove")
            if remove_btn and to_remove:
                portfolio.pop(to_remove, None)
                save_portfolio(portfolio)
                st.experimental_rerun()

    st.divider()

    rate = st.number_input("Growth Rate %", value=10.0)
    years = st.number_input("Years Ahead", value=5, step=1)

    df, total_value, total_cost, dividends, next_div_date, next_div_amount = get_portfolio_df(portfolio)
    gain = total_value - total_cost
    roi = (gain / total_cost * 100) if total_cost else 0

    sector_df = get_sector_breakdown(df) if not df.empty else pd.DataFrame()
    hist_df = simulate_historical_growth(df)
    future_df = simulate_future_growth(total_value, years or 1, rate / 100)

    growth_df = pd.concat([
        hist_df.assign(Type="Historical"),
        future_df.assign(Type="Projected"),
    ])
    growth_df.sort_values("Date", inplace=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Value", f"${total_value:,.2f}")
    col1.write(f"${total_cost:,.2f} invested")
    col2.metric("Total Profit", f"{gain:+,.2f}")
    col3.metric("ROI", f"{roi:.2f}%")
    col4.metric("Passive Income", f"${dividends:,.2f}")

    future_value = future_df["Value"].iloc[-1] if not future_df.empty else total_value
    future_date = future_df["Date"].iloc[-1].strftime("%Y-%m-%d") if not future_df.empty else CURRENT_DATE.strftime("%Y-%m-%d")
    c1, c2, c3 = st.columns(3)
    c1.metric("Projected Value", f"${future_value:,.2f}", help=f"as of {future_date}")
    next_amt_text = f"${next_div_amount:,.2f}" if next_div_amount else "N/A"
    next_date_text = next_div_date.strftime("%Y-%m-%d") if next_div_date else "N/A"
    c2.metric("Next Dividend", next_amt_text, help=next_date_text)

    beta_portfolio = 0.0
    if not df.empty:
        beta_portfolio = (df["Beta"] * df["Value"]).sum() / df["Value"].sum()
    c3.metric("Portfolio Beta", f"{beta_portfolio:.2f}")

    if not df.empty:
        st.subheader("Stock Performance")
        selected = st.selectbox("Ticker", df["Ticker"].tolist())
        if selected:
            hist = fetch_stock(selected).history(period="1y")
            if not hist.empty:
                hist_fig = px.line(hist, x=hist.index, y="Close", title=f"{selected} 1Y Price")
                st.plotly_chart(hist_fig, use_container_width=True)

    if not df.empty:
        pie_fig = px.pie(df, names="Ticker", values="Value", hole=0.5)
        sector_fig = px.pie(sector_df, names="Sector", values="Value", hole=0.3, title="Sector Allocation")
        growth_fig = px.line(growth_df, x="Date", y="Value", color="Type", title="Portfolio Growth Projection")
        gain_bar = px.bar(df, x="Ticker", y="Gain", title="Gain by Ticker")

        st.plotly_chart(pie_fig, use_container_width=True)
        st.plotly_chart(sector_fig, use_container_width=True)
        st.plotly_chart(growth_fig, use_container_width=True)
        st.plotly_chart(gain_bar, use_container_width=True)

        display_cols = [
            "Ticker",
            "Shares",
            "Cost Basis",
            "Price",
            "Value",
            "Invested",
            "Gain",
            "Allocation",
            "Dividend Yield %",
            "Dividend Income",
            "Next Dividend Date",
            "Price Target",
            "PE Ratio",
            "52w Low",
            "52w High",
            "Beta",
            "1Y Change %",
        ]
        st.dataframe(df[display_cols].round(2), use_container_width=True)
    else:
        st.info("Add some positions to get started")


if __name__ == "__main__":
    main()
