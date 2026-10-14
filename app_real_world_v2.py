"""
Sales Forecasting for a Kenyan Baby Products Retailer
----------------------------------------------------
Run:
    pip install streamlit pandas numpy matplotlib statsmodels sqlalchemy "psycopg[binary]"
    streamlit run app.py

Expected project structure:
    app.py
    artifacts/
        forecast_config.json
        monthly_net_sales.csv

The artifacts are created by running the Deployment cells in the project notebook.
If the artifacts are not available, upload a monthly CSV containing a date/month column
and a net_sales column. Forecasts are generated from monthly sales history.
"""

from pathlib import Path
import json
import warnings
import logging

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from sqlalchemy import (
    create_engine, MetaData, Table, Column, String, Integer, Float, DateTime,
    ForeignKey, select, update, insert, text, func
)

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Application log: configure a writable path via APP_LOG_FILE in deployment if needed.
import os
LOG_PATH = os.getenv("APP_LOG_FILE", str(Path(__file__).resolve().parent / "app.log"))
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("baby_retail_app")


# -------------------------------------------------------------------
# App configuration and project constants
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Baby Retail Sales Forecast",
    page_icon="🍼",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_TITLE = "Sales Forecasting for a Kenyan Baby Products Retailer"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
CONFIG_PATH = ARTIFACT_DIR / "forecast_config.json"
HISTORY_PATH = ARTIFACT_DIR / "monthly_net_sales.csv"

MODEL_SPECS = {
    "Seasonal-Naive": {"kind": "snaive"},
    "Seasonal-Naive + YoY growth": {"kind": "snaive_growth", "window": 3},
    "Holt-Winters (additive)": {
        "kind": "hw", "trend": "add", "damped": False, "seasonal": "add"
    },
    "Holt-Winters (damped trend)": {
        "kind": "hw", "trend": "add", "damped": True, "seasonal": "add"
    },
    "Holt (trend, no seasonality)": {
        "kind": "hw", "trend": "add", "damped": False, "seasonal": None
    },
    "SARIMA(1,1,1)(1,1,0,12)": {
        "kind": "sarima", "order": (1, 1, 1), "seasonal_order": (1, 1, 0, 12)
    },
    "SARIMA(0,1,1)(0,1,1,12)": {
        "kind": "sarima", "order": (0, 1, 1), "seasonal_order": (0, 1, 1, 12)
    },
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def load_config():
    """Load the configuration saved by the notebook, if available."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def load_history_from_artifact():
    """Load the monthly history exported by the notebook."""
    if not HISTORY_PATH.exists():
        return None
    try:
        frame = pd.read_csv(HISTORY_PATH)
        if frame.empty:
            return None

        # Notebook CSV normally stores the month index in the first column.
        date_col = next(
            (c for c in frame.columns if str(c).strip().lower() in
             {"month", "date", "datetime", "unnamed: 0", "index"}),
            frame.columns[0],
        )
        target_col = "net_sales" if "net_sales" in frame.columns else None
        if target_col is None:
            target_col = next(
                (c for c in frame.columns if str(c).strip().lower() in
                 {"sales", "monthly_sales", "value", "net sales"}),
                None,
            )
        if target_col is None:
            return None

        return prepare_history(frame[date_col], frame[target_col])
    except Exception:
        return None


def prepare_history(dates, values):
    """Validate and normalize monthly history to month-start frequency."""
    dates = pd.to_datetime(dates, errors="coerce")
    values = pd.to_numeric(values, errors="coerce")
    frame = pd.DataFrame({"month": dates, "net_sales": values}).dropna()
    if frame.empty:
        raise ValueError("No valid date and sales values were found.")

    frame["month"] = frame["month"].dt.to_period("M").dt.to_timestamp()
    # If the input has more than one row per month, sum sales to monthly totals.
    frame = frame.groupby("month", as_index=False)["net_sales"].sum()
    frame = frame.sort_values("month").set_index("month")
    frame = frame.asfreq("MS")

    if frame["net_sales"].isna().any():
        missing = frame.index[frame["net_sales"].isna()].strftime("%b %Y").tolist()
        raise ValueError(
            "The monthly series has missing months: " + ", ".join(missing[:8]) +
            ". Fill or investigate missing months before forecasting."
        )
    if len(frame) < 24:
        raise ValueError(
            f"At least 24 complete monthly observations are required; found {len(frame)}."
        )
    if not np.isfinite(frame["net_sales"].to_numpy(dtype=float)).all():
        raise ValueError("Sales values must be finite numbers.")
    if (frame["net_sales"] < 0).any():
        raise ValueError("Monthly net sales contains negative values. Review refunds/adjustments before forecasting.")
    return frame["net_sales"].astype(float).rename("net_sales")


def forecast_spec(model_name, history, horizon):
    """Forecast with the selected model family used in the notebook."""
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported model in configuration: {model_name}")

    spec = MODEL_SPECS[model_name]
    y = history.astype(float).asfreq("MS")
    future_index = pd.date_range(
        y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if spec["kind"] in {"snaive", "snaive_growth"}:
            forecast = y.reindex(future_index - pd.DateOffset(years=1)).to_numpy(dtype=float)
            if spec["kind"] == "snaive_growth":
                window = int(spec.get("window", 3))
                recent = y.iloc[-window:]
                previous = y.reindex(recent.index - pd.DateOffset(years=1)).to_numpy(dtype=float)
                denominator = np.nansum(previous)
                if not np.isfinite(previous).all() or denominator == 0:
                    raise ValueError(
                        "Not enough valid year-ago observations for the growth-adjusted baseline."
                    )
                forecast = forecast * (recent.to_numpy(dtype=float).sum() / denominator)
            if not np.isfinite(forecast).all():
                raise ValueError("The seasonal-naive model needs valid values from the previous year.")
            return pd.Series(forecast, index=future_index, name="forecast")

        if spec["kind"] == "hw":
            fitted = ExponentialSmoothing(
                y,
                trend=spec["trend"],
                damped_trend=bool(spec["damped"]),
                seasonal=spec["seasonal"],
                seasonal_periods=12 if spec["seasonal"] else None,
                initialization_method="estimated",
            ).fit()
            forecast = fitted.forecast(horizon)

        elif spec["kind"] == "sarima":
            fitted = SARIMAX(
                y,
                order=tuple(spec["order"]),
                seasonal_order=tuple(spec["seasonal_order"]),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            forecast = fitted.forecast(horizon)
        else:
            raise ValueError(f"Unknown model type: {spec['kind']}")

    return pd.Series(np.asarray(forecast, dtype=float), index=future_index, name="forecast")


def wape_percent(actual, predicted):
    """Calculate WAPE as a percentage."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual).sum()
    if denominator == 0:
        return np.nan
    return float(np.abs(actual - predicted).sum() / denominator * 100)


def format_kes(value, compact=False):
    """Format a numeric amount as Kenyan shillings."""
    if pd.isna(value):
        return "—"
    if compact and abs(value) >= 1_000_000:
        return f"KES {value / 1_000_000:,.2f}M"
    return f"KES {value:,.0f}"


# -------------------------------------------------------------------
# Persistent inventory database
# -------------------------------------------------------------------
def get_database_url():
    """Use Neon/PostgreSQL when configured; otherwise use local SQLite for development."""
    try:
        secret_url = st.secrets.get("DATABASE_URL", "")
    except Exception:
        secret_url = ""
    import os
    url = os.getenv("DATABASE_URL") or secret_url
    if url:
        # Some providers supply postgres://, while SQLAlchemy expects postgresql://.
        if url.startswith("postgres://"):
            url = "postgresql+psycopg://" + url[len("postgres://"):]
        elif url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        return url, True
    return "sqlite:///baby_retail_inventory.db", False


@st.cache_resource
def get_inventory_db():
    url, is_remote = get_database_url()
    engine = create_engine(url, pool_pre_ping=True, future=True)
    metadata = MetaData()
    products = Table(
        "inventory_products", metadata,
        Column("sku", String(100), primary_key=True),
        Column("product_name", String(255), nullable=False),
        Column("current_stock", Integer, nullable=False, default=0),
        Column("avg_monthly_units", Float, nullable=False, default=0),
        Column("lead_time_days", Integer, nullable=False, default=0),
        Column("unit_cost_kes", Float, nullable=False, default=0),
        Column("safety_stock_units", Integer, nullable=False, default=0),
        Column("min_order_qty", Integer, nullable=False, default=1),
        Column("updated_at", DateTime, nullable=False, server_default=func.now(), onupdate=func.now()),
    )
    movements = Table(
        "inventory_movements", metadata,
        Column("movement_id", Integer, primary_key=True, autoincrement=True),
        Column("sku", String(100), ForeignKey("inventory_products.sku"), nullable=False),
        Column("movement_type", String(30), nullable=False),
        Column("quantity_delta", Integer, nullable=False),
        Column("reason", String(500), nullable=False),
        Column("reference", String(150)),
        Column("created_at", DateTime, nullable=False, server_default=func.now()),
    )
    business_events = Table(
        "business_events", metadata,
        Column("event_id", Integer, primary_key=True, autoincrement=True),
        Column("event_date", String(10), nullable=False),
        Column("event_type", String(40), nullable=False),
        Column("sku", String(100), ForeignKey("inventory_products.sku")),
        Column("supplier_name", String(255)),
        Column("numeric_value", Float),
        Column("reference", String(150)),
        Column("notes", String(1000), nullable=False),
        Column("created_at", DateTime, nullable=False, server_default=func.now()),
    )
    forecast_reviews = Table(
        "forecast_reviews", metadata,
        Column("review_id", Integer, primary_key=True, autoincrement=True),
        Column("forecast_month", String(7), nullable=False),
        Column("forecast_kes", Float, nullable=False),
        Column("actual_kes", Float, nullable=False),
        Column("absolute_error_kes", Float, nullable=False),
        Column("absolute_percentage_error", Float),
        Column("notes", String(1000)),
        Column("created_at", DateTime, nullable=False, server_default=func.now()),
    )
    metadata.create_all(engine)
    return engine, products, movements, business_events, forecast_reviews, is_remote


def read_inventory_table(engine, table):
    with engine.connect() as conn:
        rows = conn.execute(select(table)).mappings().all()
    return pd.DataFrame([dict(row) for row in rows])


def upsert_inventory_product(conn, products, row, update_existing_stock=False):
    existing = conn.execute(select(products.c.sku).where(products.c.sku == row["sku"])).first()
    values = {
        "sku": str(row["sku"]).strip(),
        "product_name": str(row["product_name"]).strip(),
        "avg_monthly_units": float(row["avg_monthly_units"]),
        "lead_time_days": int(row["lead_time_days"]),
        "unit_cost_kes": float(row["unit_cost_kes"]),
        "safety_stock_units": int(row.get("safety_stock_units", 0)),
        "min_order_qty": max(1, int(row.get("min_order_qty", 1))),
        "updated_at": datetime.utcnow(),
    }
    if existing:
        if update_existing_stock:
            values["current_stock"] = int(row["current_stock"])
        conn.execute(update(products).where(products.c.sku == values["sku"]).values(**values))
    else:
        values["current_stock"] = int(row["current_stock"])
        conn.execute(insert(products).values(**values))


# -------------------------------------------------------------------
# Header and sidebar
# -------------------------------------------------------------------
st.title("🍼 Sales Forecasting for a Kenyan Baby Products Retailer")
st.caption(
    "A monthly sales planning dashboard for forecasting revenue, reviewing historical "
    "performance, and supporting stock and cash-flow decisions."
)

config = load_config()
artifact_history = load_history_from_artifact()

with st.sidebar:
    st.header("Forecast settings")
    st.write("**Forecast horizon**")
    horizon = st.selectbox("Months ahead", options=[3, 1, 2, 4, 6, 12], index=0)

    st.divider()
    st.subheader("Data source")
    st.caption(
        "The app first looks for the monthly history exported by the notebook. "
        "You can upload another monthly CSV if needed."
    )
    uploaded_file = st.file_uploader(
        "Upload monthly sales CSV",
        type=["csv"],
        help="Include a month/date column and a net_sales or sales column.",
    )


# -------------------------------------------------------------------
# Load history from artifact or uploaded CSV
# -------------------------------------------------------------------
history = None
data_source_label = ""

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        normalized_columns = {str(c).strip().lower(): c for c in uploaded_df.columns}

        date_col = next(
            (normalized_columns[c] for c in
             ["month", "date", "datetime", "timestamp", "period"]
             if c in normalized_columns),
            None,
        )
        sales_col = next(
            (normalized_columns[c] for c in
             ["net_sales", "sales", "monthly_sales", "net sales"]
             if c in normalized_columns),
            None,
        )

        if date_col is None or sales_col is None:
            st.error(
                "CSV columns were not recognized. Include a month/date column and a "
                "net_sales (or sales) column."
            )
            st.stop()

        history = prepare_history(uploaded_df[date_col], uploaded_df[sales_col])
        data_source_label = f"Uploaded file: {uploaded_file.name}"
    except Exception as exc:
        st.error(f"Could not load the uploaded CSV: {exc}")
        st.stop()
elif artifact_history is not None:
    history = artifact_history
    data_source_label = str(HISTORY_PATH.relative_to(Path(__file__).resolve().parent))
else:
    st.info(
        "No notebook artifacts were found. Run the Deployment cells in your notebook "
        "to create `artifacts/monthly_net_sales.csv` and `artifacts/forecast_config.json`, "
        "or upload a monthly sales CSV from the sidebar."
    )
    st.markdown(
        """
        **Expected CSV example**

        | month | net_sales |
        |---|---:|
        | 2021-01-01 | 1250000 |
        | 2021-02-01 | 1320000 |
        | 2021-03-01 | 1290000 |
        """
    )
    st.stop()


# -------------------------------------------------------------------
# Model selection and forecast
# -------------------------------------------------------------------
saved_model = config.get("selected_model", "Holt-Winters (damped trend)")
available_models = list(MODEL_SPECS.keys())
default_model = saved_model if saved_model in available_models else "Holt-Winters (damped trend)"

with st.sidebar:
    model_name = st.selectbox(
        "Forecasting model",
        options=available_models,
        index=available_models.index(default_model),
        help=(
            "The saved notebook model is selected by default when its configuration exists. "
            "Changing this setting lets you explore other candidate models."
        ),
    )

try:
    forecast = forecast_spec(model_name, history, int(horizon))
    logger.info(
        "Forecast generated model=%s horizon=%s observations=%s latest_month=%s",
        model_name, int(horizon), len(history), history.index[-1].strftime("%Y-%m")
    )
except Exception as exc:
    logger.exception("Forecast generation failed")
    st.error(f"Forecast could not be generated: {exc}")
    st.stop()

# Use the notebook's validation-derived interval when available.
interval_info = config.get("interval", {})
half_width = interval_info.get("half_width_pct", None)
if half_width is None or not np.isfinite(float(half_width)):
    # Conservative fallback for exploratory forecasts only; clearly labelled below.
    half_width = 0.15
    interval_source = "Illustrative ±15% range (fallback; not calibrated from validation)"
else:
    half_width = float(half_width)
    interval_source = (
        f"Approximate {float(interval_info.get('level', 0.80)):.0%} range based on "
        "the notebook's rolling-origin validation errors"
    )

forecast_df = pd.DataFrame({
    "Forecast (KES)": forecast,
    "Lower range (KES)": forecast * (1 - half_width),
    "Upper range (KES)": forecast * (1 + half_width),
})
forecast_df.index.name = "Month"


# -------------------------------------------------------------------
# Overview metrics
# -------------------------------------------------------------------
total_forecast = float(forecast.sum())
last_month_sales = float(history.iloc[-1])
average_recent = float(history.tail(3).mean())
first_forecast = float(forecast.iloc[0])
change_vs_last = (first_forecast / last_month_sales - 1) * 100 if last_month_sales else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest actual month", history.index[-1].strftime("%b %Y"))
c2.metric("Latest monthly sales", format_kes(last_month_sales, compact=True))
c3.metric(f"Forecast total ({horizon} months)", format_kes(total_forecast, compact=True))
c4.metric("First forecast vs latest month", f"{change_vs_last:+.1f}%" if np.isfinite(change_vs_last) else "—")

st.caption(f"Data source: {data_source_label} · {len(history)} monthly observations · Model: {model_name}")


# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab_forecast, tab_history, tab_inventory, tab_operations, tab_scenarios, tab_evaluation, tab_guidance = st.tabs(
    ["Forecast", "Historical Sales", "Inventory Planner", "Business Events & Monitoring", "Business Scenarios", "Evaluation & Model Info", "Business Guidance"]
)

with tab_forecast:
    st.subheader(f"Forecast for the next {horizon} months")
    st.caption(interval_source)

    display_forecast = forecast_df.copy()
    display_forecast.index = display_forecast.index.strftime("%b %Y")
    st.dataframe(
        display_forecast.style.format({
            "Forecast (KES)": "KES {:,.0f}",
            "Lower range (KES)": "KES {:,.0f}",
            "Upper range (KES)": "KES {:,.0f}",
        }),
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    recent = history.tail(18)
    ax.plot(recent.index, recent.values, marker="o", label="Actual monthly sales")
    ax.plot(forecast.index, forecast.values, marker="o", linestyle="--", label="Forecast")
    ax.fill_between(
        forecast.index,
        forecast.values * (1 - half_width),
        forecast.values * (1 + half_width),
        alpha=0.18,
        label="Approximate forecast range",
    )
    ax.set_title("Historical Net Sales and Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Net Sales (KES)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Give the forecast index a predictable lowercase name before resetting it.
    # forecast_df's index is named "Month" (capital M), so reset_index() would
    # otherwise create a "Month" column and cause KeyError: 'month'.
    download_df = forecast_df.copy().rename_axis("month").reset_index()
    download_df["month"] = pd.to_datetime(
        download_df["month"], errors="coerce"
    ).dt.strftime("%Y-%m")
    st.download_button(
        "Download forecast CSV",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name="baby_retail_sales_forecast.csv",
        mime="text/csv",
    )

    st.info(
        "The forecast estimates revenue, not units or profit. Use it alongside supplier lead "
        "times, stock-on-hand, promotions, margins, and known stock-outs before making orders."
    )

with tab_history:
    st.subheader("Historical monthly net sales")
    left, right = st.columns(2)
    left.metric("History starts", history.index.min().strftime("%b %Y"))
    right.metric("History ends", history.index.max().strftime("%b %Y"))

    st.line_chart(history.rename("Net sales (KES)"), use_container_width=True)

    monthly_summary = pd.DataFrame({
        "Net sales (KES)": history,
        "3-month rolling average (KES)": history.rolling(3).mean(),
        "Year": history.index.year,
    })
    st.dataframe(
        monthly_summary.style.format({
            "Net sales (KES)": "KES {:,.0f}",
            "3-month rolling average (KES)": "KES {:,.0f}",
        }),
        use_container_width=True,
    )

    st.download_button(
        "Download historical sales CSV",
        data=history.rename("net_sales").to_csv().encode("utf-8"),
        file_name="monthly_net_sales_history.csv",
        mime="text/csv",
    )


with tab_inventory:
    st.subheader("Inventory operations and reorder planning")
    st.write(
        "Maintain a product catalogue, record stock changes, review reorder alerts, "
        "and export a suggested purchase plan. Inventory balances are saved in a database."
    )
    try:
        engine, products_table, movements_table, events_table, forecast_reviews_table, is_remote_db = get_inventory_db()
        if is_remote_db:
            st.success("Inventory database: PostgreSQL/Neon connection configured.")
        else:
            st.warning(
                "Inventory database: local SQLite file. This is suitable for local testing on one computer, "
                "but use Neon/PostgreSQL for shared or cloud production use."
            )

        with st.expander("1. Import or update product catalogue", expanded=True):
            st.caption(
                "Required CSV columns: sku, product_name, current_stock, avg_monthly_units, "
                "lead_time_days, unit_cost_kes. Optional: safety_stock_units, min_order_qty."
            )
            inventory_file = st.file_uploader(
                "Upload inventory CSV", type=["csv"], key="inventory_csv",
                help="Use one row per SKU. Stock and demand must be in units; cost must be in KES."
            )
            st.caption("For existing SKUs, imports update product details but preserve saved stock balances. Record stock changes in the movement form below so they appear in the audit trail.")
            if inventory_file is not None and st.button("Import product data", type="primary"):
                try:
                    incoming = pd.read_csv(inventory_file)
                    incoming.columns = [str(c).strip().lower() for c in incoming.columns]
                    required = ["sku", "product_name", "current_stock", "avg_monthly_units", "lead_time_days", "unit_cost_kes"]
                    missing = [c for c in required if c not in incoming.columns]
                    if missing:
                        st.error("Missing columns: " + ", ".join(missing))
                    elif incoming.empty:
                        st.error("The uploaded CSV has no product rows.")
                    else:
                        for col, default in [("safety_stock_units", 0), ("min_order_qty", 1)]:
                            if col not in incoming.columns:
                                incoming[col] = default
                        numeric = ["current_stock", "avg_monthly_units", "lead_time_days", "unit_cost_kes", "safety_stock_units", "min_order_qty"]
                        for col in numeric:
                            incoming[col] = pd.to_numeric(incoming[col], errors="coerce")
                        bad = incoming[numeric].isna().any(axis=1)
                        bad |= (incoming[numeric] < 0).any(axis=1)
                        for whole_col in ["current_stock", "lead_time_days", "safety_stock_units", "min_order_qty"]:
                            bad |= incoming[whole_col].notna() & (incoming[whole_col] % 1 != 0)
                        bad |= incoming["sku"].isna() | incoming["product_name"].isna()
                        duplicate = incoming["sku"].astype(str).str.strip().duplicated(keep=False)
                        if bad.any():
                            st.error(f"{int(bad.sum())} row(s) contain missing or negative values. Correct the file and retry.")
                        elif duplicate.any():
                            st.error("Duplicate SKU codes found. Each product must have one unique SKU row.")
                        else:
                            with engine.begin() as conn:
                                for _, row in incoming.iterrows():
                                    upsert_inventory_product(conn, products_table, row, update_existing_stock=False)
                            st.success(f"Imported/updated {len(incoming)} product rows. Changes are saved in the database.")
                            st.rerun()
                except Exception as exc:
                    st.error(f"Import failed: {exc}")

        inventory = read_inventory_table(engine, products_table)
        if inventory.empty:
            st.info("No products are saved yet. Import your product CSV to get started.")
        else:
            inventory["sku"] = inventory["sku"].astype(str)
            st.subheader("2. Current stock and stock movement")
            st.caption("Use this section to record sales, received stock, damaged goods, or a verified stock correction.")
            options = inventory["sku"].tolist()
            with st.form("stock_movement_form"):
                chosen_sku = st.selectbox("Product SKU", options=options, format_func=lambda sku: f"{sku} — {inventory.loc[inventory['sku'] == sku, 'product_name'].iloc[0]}")
                movement_type = st.selectbox("Movement type", ["Sale / stock issued", "Stock received", "Damaged / expired", "Stock correction"])
                quantity = st.number_input("Quantity (whole units; for stock correction, enter the verified new balance)", min_value=0, value=1, step=1)
                reason = st.text_input("Reason / note", placeholder="e.g. Customer sale, supplier delivery, damaged pack")
                reference = st.text_input("Reference (optional)", placeholder="e.g. receipt number or invoice")
                submitted = st.form_submit_button("Save stock movement")
            if submitted:
                if not reason.strip():
                    st.error("Enter a reason so the stock change can be audited.")
                elif movement_type != "Stock correction" and int(quantity) < 1:
                    st.error("Quantity must be at least 1 for sales, receipts, or damaged/expired stock.")
                else:
                    product_row = inventory.loc[inventory["sku"] == chosen_sku].iloc[0]
                    qty = int(quantity)
                    if movement_type in ["Sale / stock issued", "Damaged / expired"]:
                        delta = -qty
                    elif movement_type == "Stock received":
                        delta = qty
                    else:
                        # A correction means set the balance to the entered quantity.
                        delta = qty - int(product_row["current_stock"])
                    new_stock = int(product_row["current_stock"]) + delta
                    if new_stock < 0:
                        st.error(f"Cannot save this movement: available stock is {int(product_row['current_stock'])} units.")
                    else:
                        with engine.begin() as conn:
                            conn.execute(update(products_table).where(products_table.c.sku == chosen_sku).values(current_stock=new_stock, updated_at=datetime.utcnow()))
                            conn.execute(insert(movements_table).values(
                                sku=chosen_sku, movement_type=movement_type, quantity_delta=delta,
                                reason=reason.strip(), reference=reference.strip() or None,
                                created_at=datetime.utcnow()
                            ))
                        st.success(f"Stock updated. New balance: {new_stock} units.")
                        st.rerun()

            inventory = read_inventory_table(engine, products_table)
            inventory["avg_daily_units"] = inventory["avg_monthly_units"] / 30.4375
            review_days = st.slider(
                "Stock review period (days)", min_value=7, max_value=60, value=30, step=1,
                help="How many days of expected demand to cover beyond supplier lead time."
            )
            inventory["reorder_point_units"] = np.ceil(
                inventory["avg_daily_units"] * inventory["lead_time_days"] + inventory["safety_stock_units"]
            ).astype(int)
            inventory["target_stock_units"] = np.ceil(
                inventory["avg_daily_units"] * (inventory["lead_time_days"] + review_days) + inventory["safety_stock_units"]
            ).astype(int)
            inventory["suggested_order_units"] = np.maximum(
                0, inventory["target_stock_units"] - inventory["current_stock"]
            ).astype(int)
            inventory["suggested_order_units"] = inventory.apply(
                lambda row: int(np.ceil(row["suggested_order_units"] / max(1, int(row["min_order_qty"]))) * max(1, int(row["min_order_qty"]))) if row["suggested_order_units"] > 0 else 0,
                axis=1
            )
            inventory["estimated_purchase_cost_kes"] = inventory["suggested_order_units"] * inventory["unit_cost_kes"]
            inventory["days_of_stock_left"] = np.where(
                inventory["avg_daily_units"] > 0,
                inventory["current_stock"] / inventory["avg_daily_units"], np.inf
            )
            inventory["stock_status"] = np.where(
                inventory["current_stock"] <= inventory["reorder_point_units"], "REORDER NOW",
                np.where(inventory["current_stock"] <= inventory["target_stock_units"], "WATCH", "Stock OK")
            )

            total_cost = float(inventory["estimated_purchase_cost_kes"].sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Products tracked", f"{len(inventory):,}")
            c2.metric("Reorder now", f"{int((inventory['stock_status'] == 'REORDER NOW').sum()):,}")
            c3.metric("Products to order", f"{int((inventory['suggested_order_units'] > 0).sum()):,}")
            c4.metric("Estimated order cost", format_kes(total_cost, compact=True))

            budget = st.number_input("Available purchasing budget (KES)", min_value=0.0, value=total_cost, step=1000.0, key="inventory_budget")
            if total_cost > budget:
                st.warning(f"Order plan exceeds budget by {format_kes(total_cost - budget)}. Prioritize urgent items and verify supplier terms.")
            else:
                st.info(f"Order plan is within budget by {format_kes(budget - total_cost)}.")

            display_cols = ["sku", "product_name", "current_stock", "avg_monthly_units", "lead_time_days", "days_of_stock_left", "reorder_point_units", "target_stock_units", "suggested_order_units", "unit_cost_kes", "estimated_purchase_cost_kes", "stock_status"]
            st.dataframe(inventory[display_cols].style.format({
                "days_of_stock_left": lambda x: "No recent demand" if not np.isfinite(x) else f"{x:.1f} days",
                "unit_cost_kes": "KES {:,.2f}", "estimated_purchase_cost_kes": "KES {:,.2f}"
            }), use_container_width=True, hide_index=True)
            st.subheader("Human review before ordering")
            st.caption(
                "This is a suggested plan, not a purchase order. Confirm stock counts, demand, "
                "supplier lead times, pack sizes, prices, and budget before exporting."
            )
            reviewed = st.checkbox(
                "I have reviewed the quantities, supplier details, costs, and available budget.",
                key="purchase_plan_reviewed"
            )
            if reviewed:
                st.download_button(
                    "Export reviewed purchase plan CSV",
                    data=inventory[display_cols].to_csv(index=False).encode("utf-8"),
                    file_name=f"reviewed_replenishment_plan_{datetime.now():%Y%m%d}.csv",
                    mime="text/csv", key="inventory_plan_download"
                )
            else:
                st.info("Review confirmation is required before the purchase plan can be exported.")

            st.subheader("3. Stock movement audit trail")
            movements = read_inventory_table(engine, movements_table)
            if not movements.empty:
                movements = movements.sort_values("created_at", ascending=False).head(100)
                st.dataframe(movements, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download stock movement history CSV",
                    data=movements.to_csv(index=False).encode("utf-8"),
                    file_name="inventory_movement_history.csv", mime="text/csv",
                    key="movement_history_download"
                )
            else:
                st.caption("No stock movements have been recorded yet.")

            st.caption(
                "Operational safeguards: verify physical stock counts, supplier lead times, minimum order quantities, "
                "and costs before ordering. Demand currently uses average monthly units, not a SKU-level ML forecast. "
                "This app does not yet process barcode scans, supplier APIs, payment systems, or multi-warehouse transfers."
            )
    except Exception as exc:
        st.error(f"Inventory database could not be opened: {exc}")
        st.info("For local use, install the requirements file. For shared deployment, configure DATABASE_URL to a PostgreSQL/Neon connection string.")


with tab_operations:
    st.subheader("Business events, supplier performance and forecast monitoring")
    st.caption(
        "Record operational events as structured data instead of relying only on free-text notes. "
        "These records support future forecast improvements and audit reviews."
    )
    try:
        engine, products_table, movements_table, events_table, forecast_reviews_table, is_remote_db = get_inventory_db()
        products_for_events = read_inventory_table(engine, products_table)
        sku_options = ["(No SKU / business-wide)"]
        if not products_for_events.empty:
            sku_options += products_for_events["sku"].astype(str).tolist()

        with st.expander("Record a business event", expanded=True):
            with st.form("business_event_form"):
                event_date = st.date_input("Event date", value=datetime.now().date())
                event_type = st.selectbox(
                    "Event type",
                    [
                        "Stock-out", "Promotion", "Price change", "Supplier delivery",
                        "Supplier delay", "Data correction", "Other"
                    ]
                )
                event_sku = st.selectbox("Related SKU (if applicable)", sku_options)
                supplier_name = st.text_input("Supplier name (if applicable)")
                numeric_value = st.number_input(
                    "Numeric value (optional)", min_value=0.0, value=0.0, step=1.0,
                    help="Examples: new selling price, promotion discount percentage, units delivered, or delay days. Explain the meaning in notes."
                )
                event_reference = st.text_input("Reference (optional)", placeholder="Invoice, delivery note, promotion ID")
                event_notes = st.text_area(
                    "Details / reason",
                    placeholder="Describe what happened, affected dates, and any evidence or corrective action."
                )
                save_event = st.form_submit_button("Save business event")
            if save_event:
                if not event_notes.strip():
                    st.error("Please provide details so the event can be understood later.")
                else:
                    try:
                        with engine.begin() as conn:
                            conn.execute(insert(events_table).values(
                                event_date=event_date.isoformat(),
                                event_type=event_type,
                                sku=None if event_sku == "(No SKU / business-wide)" else event_sku,
                                supplier_name=supplier_name.strip() or None,
                                numeric_value=float(numeric_value) if numeric_value > 0 else None,
                                reference=event_reference.strip() or None,
                                notes=event_notes.strip(),
                                created_at=datetime.utcnow(),
                            ))
                        logger.info("Business event recorded type=%s sku=%s", event_type, event_sku)
                        st.success("Business event saved.")
                        st.rerun()
                    except Exception:
                        logger.exception("Could not save business event")
                        st.error("The event could not be saved. Check the application log and database connection.")

        events = read_inventory_table(engine, events_table)
        st.subheader("Business event history")
        if events.empty:
            st.info("No business events have been recorded yet.")
        else:
            events = events.sort_values("created_at", ascending=False)
            st.dataframe(events, use_container_width=True, hide_index=True)
            st.download_button(
                "Download business event log",
                data=events.to_csv(index=False).encode("utf-8"),
                file_name="business_event_log.csv",
                mime="text/csv",
                key="business_event_download",
            )

        st.divider()
        st.subheader("Compare a completed month's forecast with actual sales")
        st.caption("Enter the forecast that was originally issued for a month, then compare it with the final actual net sales.")
        with st.form("forecast_review_form"):
            review_month = st.text_input("Forecast month (YYYY-MM)", placeholder="2026-09")
            reviewed_forecast = st.number_input("Forecast net sales (KES)", min_value=0.0, value=0.0, step=1000.0)
            actual_sales = st.number_input("Actual net sales (KES)", min_value=0.0, value=0.0, step=1000.0)
            review_notes = st.text_input("Review notes (optional)", placeholder="Explain a promotion, stock-out, or unusual event")
            save_review = st.form_submit_button("Save forecast review")
        if save_review:
            valid_month = (
                len(review_month) == 7 and review_month[4:5] == "-"
                and review_month[:4].isdigit() and review_month[5:7].isdigit()
                and 1 <= int(review_month[5:7]) <= 12
            )
            if not valid_month:
                st.error("Enter the month in YYYY-MM format, for example 2026-09.")
            elif reviewed_forecast <= 0:
                st.error("Forecast sales must be greater than zero to calculate percentage error.")
            else:
                absolute_error = abs(actual_sales - reviewed_forecast)
                ape = absolute_error / abs(actual_sales) * 100 if actual_sales > 0 else None
                try:
                    with engine.begin() as conn:
                        conn.execute(insert(forecast_reviews_table).values(
                            forecast_month=review_month,
                            forecast_kes=float(reviewed_forecast),
                            actual_kes=float(actual_sales),
                            absolute_error_kes=float(absolute_error),
                            absolute_percentage_error=float(ape) if ape is not None else None,
                            notes=review_notes.strip() or None,
                            created_at=datetime.utcnow(),
                        ))
                    logger.info("Forecast review saved month=%s", review_month)
                    st.success(f"Review saved. Absolute error: {format_kes(absolute_error)}.")
                    st.rerun()
                except Exception:
                    logger.exception("Could not save forecast review")
                    st.error("The review could not be saved. Check the application log and database connection.")

        reviews = read_inventory_table(engine, forecast_reviews_table)
        st.subheader("Forecast monitoring history")
        if reviews.empty:
            st.info("No forecast-versus-actual reviews have been recorded yet.")
        else:
            reviews = reviews.sort_values("forecast_month", ascending=False)
            st.dataframe(reviews, use_container_width=True, hide_index=True)
            valid_ape = pd.to_numeric(reviews["absolute_percentage_error"], errors="coerce").dropna()
            if not valid_ape.empty:
                st.metric("Mean absolute percentage error (reviewed months)", f"{valid_ape.mean():.1f}%")
            st.download_button(
                "Download forecast monitoring history",
                data=reviews.to_csv(index=False).encode("utf-8"),
                file_name="forecast_monitoring_history.csv",
                mime="text/csv",
                key="forecast_reviews_download",
            )
    except Exception:
        logger.exception("Business operations tab failed to load")
        st.error("Business events or forecast monitoring could not be loaded. Check the database connection and application log.")

with tab_scenarios:
    st.subheader("Business scenarios and sales planning")
    st.write(
        "Explore how a change in expected sales or gross margin could affect revenue and "
        "gross profit. These are user-defined scenarios, not model predictions."
    )
    base_revenue = float(forecast.sum())
    s1, s2, s3 = st.columns(3)
    growth_pct = s1.slider("Sales change scenario (%)", min_value=-50, max_value=100, value=0, step=5)
    margin_pct = s2.slider("Assumed gross margin (%)", min_value=0, max_value=80, value=25, step=1)
    expense_kes = s3.number_input(
        "Other operating expenses for forecast period (KES)",
        min_value=0.0, value=0.0, step=5000.0
    )
    scenario_revenue = base_revenue * (1 + growth_pct / 100)
    scenario_gross_profit = scenario_revenue * margin_pct / 100
    scenario_operating_surplus = scenario_gross_profit - expense_kes

    p1, p2, p3 = st.columns(3)
    p1.metric("Scenario revenue", format_kes(scenario_revenue, compact=True),
              delta=f"{growth_pct:+d}% vs model forecast")
    p2.metric("Estimated gross profit", format_kes(scenario_gross_profit, compact=True))
    p3.metric("Gross profit less entered expenses",
              format_kes(scenario_operating_surplus, compact=True))

    scenario_table = pd.DataFrame({
        "Scenario": ["Downside (-20%)", "Base forecast", "Upside (+20%)"],
        "Revenue (KES)": [base_revenue * 0.8, base_revenue, base_revenue * 1.2],
        "Gross profit at assumed margin (KES)": [
            base_revenue * 0.8 * margin_pct / 100,
            base_revenue * margin_pct / 100,
            base_revenue * 1.2 * margin_pct / 100
        ],
    })
    st.dataframe(
        scenario_table.style.format({
            "Revenue (KES)": "KES {:,.0f}",
            "Gross profit at assumed margin (KES)": "KES {:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    st.download_button(
        "Download scenario summary CSV",
        data=scenario_table.to_csv(index=False).encode("utf-8"),
        file_name="business_sales_scenarios.csv",
        mime="text/csv",
        key="scenario_download"
    )
    st.warning(
        "Gross margin and expenses must come from your actual records. This simplified scenario "
        "does not include tax, debt repayments, supplier payment timing, or other cash movements."
    )

with tab_evaluation:
    st.subheader("Evaluation information")

    evaluation = config.get("evaluation", {})
    if evaluation:
        metric_cols = st.columns(3)
        test_wape = evaluation.get("test_wape_pct")
        base_wape = evaluation.get("baseline_test_wape_pct")
        improvement = evaluation.get("improvement_vs_baseline_pct")

        metric_cols[0].metric(
            "Held-out test WAPE",
            f"{float(test_wape):.2f}%" if test_wape is not None else "Pending",
        )
        metric_cols[1].metric(
            "Seasonal-naive WAPE",
            f"{float(base_wape):.2f}%" if base_wape is not None else "Pending",
        )
        metric_cols[2].metric(
            "Improvement vs baseline",
            f"{float(improvement):.2f}%" if improvement is not None else "Pending",
        )

        wape_target = float(evaluation.get("wape_target_pct", 15))
        improvement_target = float(evaluation.get("improvement_target_pct", 10))
        wape_ok = test_wape is not None and float(test_wape) <= wape_target
        improvement_ok = improvement is not None and float(improvement) >= improvement_target

        st.write(f"**WAPE target:** ≤ {wape_target:.1f}%")
        st.write(f"**Baseline improvement target:** ≥ {improvement_target:.1f}%")

        if test_wape is None or improvement is None:
            st.warning("Evaluation status is pending. Run the notebook Evaluation phase.")
        elif wape_ok and improvement_ok:
            st.success(
                "The saved model meets both predefined numerical forecasting criteria. "
                "This supports further validation; it does not by itself prove production readiness."
            )
        else:
            st.warning(
                "One or more predefined forecasting criteria were not met. Review the notebook's "
                "Evaluation phase before relying on this model for planning."
            )
    else:
        st.warning(
            "No saved evaluation metrics were found. This app can generate exploratory forecasts, "
            "but model performance cannot be confirmed until notebook evaluation artifacts are saved."
        )

    st.markdown("**Model and data limitations**")
    st.markdown(
        """
        - The notebook uses a relatively short history of monthly observations.
        - The model does not automatically account for promotions, price changes, stock-outs,
          public holidays, supplier delays, or changes in product availability.
        - Forecast ranges are approximate and are not guarantees.
        - Forecast revenue is not the same as profit, cash available, or the number of units to order.
        """
    )

with tab_guidance:
    st.subheader("Using forecasts for retailer planning")
    st.markdown(
        """
        **Inventory planning**
        - Use the central forecast as a revenue-planning reference, not as a direct unit-order quantity.
        - Combine it with stock on hand, supplier lead times, minimum order quantities, and expected prices.
        - Investigate stock-outs because observed sales can understate demand when items are unavailable.

        **Cash-flow planning**
        - Treat forecast sales as estimated revenue. Include supplier costs, operating expenses, margins,
          and payment timing separately when preparing a cash-flow plan.

        **Monthly monitoring**
        - Once each month is complete, compare actual sales with the forecast made for that month.
        - Review the model if forecast errors rise or repeatedly overestimate or underestimate sales.
        - Re-run model selection periodically as new completed months become available.
        """
    )

    st.subheader("Data quality reminder")
    st.write(
        "The source notebook notes that November 2020 is a partial month. If that observation is "
        "still present in the history, interpret it cautiously because its low total may reflect "
        "incomplete coverage rather than low underlying demand."
    )

st.divider()
st.caption(
    "Planning-support application, not yet a fully production-hardened system. Configure authentication, "
    "least-privilege database access, managed backups and restore tests before multi-user business use. "
    "Review data quality and forecast accuracy before material purchasing or financial decisions."
)
