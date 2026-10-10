"""
Sales Forecasting for a Kenyan Baby Products Retailer
----------------------------------------------------
Run:
    pip install streamlit pandas numpy matplotlib statsmodels
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

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
except Exception as exc:
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
tab_forecast, tab_history, tab_evaluation, tab_guidance = st.tabs(
    ["Forecast", "Historical Sales", "Evaluation & Model Info", "Business Guidance"]
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
    "Prototype for planning support only. Review data quality and forecast accuracy before making "
    "material purchasing or financial decisions."
)
