import pandas as pd
import numpy as np
import streamlit as st
from dataclasses import dataclass
from typing import List


from datetime import datetime
import os

LOG_FILE = "usage_log.csv"
def explain_sku_risk(row):
    bias = row["actual"] -row["forecast"]
    drivers = []

    # Demand impact
    if row["abs_error"] >= 100:
        score = row["abs_error"]  # higher error = higher impact

        if bias > 0:
            drivers.append(("Under-forecasting / stockout risk", score))
        elif bias < 0:
            drivers.append(("Over-forecasting / excess inventory risk", score))

    # Inventory impact
    if row["coverage_days"] < 10:
        score = 10 - row["coverage_days"]  # lower coverage = worse
        drivers.append(("Low inventory coverage", score))

    # Supply impact
    if row["lead_time_days"] >= 21:
        score = row["lead_time_days"]  # longer = worse
        drivers.append(("Long replenishment lead time", score))

    if row["supplier_otif"] < 0.9:
        score = (1 - row["supplier_otif"]) * 100  # lower OTIF = worse
        drivers.append(("Supplier reliability risk", score))

    # Business impact
    if row["margin"] >= 5:
        score = row["margin"]
        drivers.append(("High margin exposure", score))

    # If nothing triggered
    if not drivers:
        return ["Risk appears controlled"]

    # 🔥 NEW: sort by impact (highest first)
    drivers_sorted = sorted(drivers, key=lambda x: x[1], reverse=True)

    # Return only labels (ordered)
    return [d[0] for d in drivers_sorted[:2]]
def log_event(event_name):
    new_event = pd.DataFrame([{
        "timestamp": datetime.now(),
        "event": event_name
    }])

    if os.path.exists(LOG_FILE):
        new_event.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        new_event.to_csv(LOG_FILE, index=False)

st.set_page_config(page_title="Planning Risk Prioritiser", layout="wide")
st.markdown("""
<style>
div[data-testid="stMarkdownContainer"] .section-box {
background-color: #ffffff;
border: 1px solid #e6e6e6;
border-radius: 16px;
padding: 20px 24px;
margin-bottom: 20px;
box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

if "visited" not in st.session_state:
    log_event("app_opened")
    st.session_state["visited"]=True


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .main-title {font-size: 2.5rem; font-weight: 800; margin-bottom: 0.2rem;}
    .subtitle {color: #5f6368; margin-bottom: 1.25rem;}
    .metric-card {
        background: #f8f9fb;
        border: 1px solid #eceff3;
        border-radius: 14px;
        padding: 0.9rem 1rem;
    }
    .section-note {
        color: #5f6368;
        font-size: 0.95rem;
        margin-top: -0.25rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED_COLUMNS = [
    "sku",
    "forecast",
    "actual",
    "volume",
    "margin",
    "inventory_on_hand",
    "safety_stock",
    "lead_time_days",
    "supplier_otif",
]
OPTIONAL_HISTORY_COLUMNS = ["hist_1", "hist_2", "hist_3"]
ACTION_ICON = {
    "EXPEDITE": "🔴",
    "SUPPLY ISSUE": "🟠",
    "REDUCE INVENTORY": "🟡",
    "MONITOR": "🟢",
}
ACTION_COLOR = {
    "EXPEDITE": "#C62828",
    "SUPPLY ISSUE": "#EF6C00",
    "REDUCE INVENTORY": "#F9A825",
    "MONITOR": "#2E7D32",
}


@dataclass
class RiskThresholds:
    high_risk_quantile: float = 0.75
    low_coverage_days: float = 14.0
    high_coverage_days: float = 45.0
    falling_demand_bias_pct: float = -0.10
    poor_otif: float = 0.85


def validate_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def safe_divide(a, b):
    b = np.where(np.asarray(b) == 0, np.nan, b)
    return np.asarray(a) / b


def compute_volatility(df: pd.DataFrame) -> pd.Series:
    hist_cols = [c for c in OPTIONAL_HISTORY_COLUMNS if c in df.columns]
    if len(hist_cols) >= 2:
        hist = df[hist_cols].astype(float)
        mean_hist = hist.mean(axis=1).replace(0, np.nan)
        cv = hist.std(axis=1, ddof=0) / mean_hist
        return cv.fillna(0)
    return pd.Series(np.zeros(len(df)), index=df.index)


def min_max_scale(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    min_v = s.min()
    max_v = s.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(np.ones(len(s)) * 0.5, index=s.index)
    return (s - min_v) / (max_v - min_v)


def format_action(action: str) -> str:
    return f"{ACTION_ICON.get(action, '⚪')} {action}"


def compute_metrics(df: pd.DataFrame, demand_weight: float, supply_weight: float, inventory_weight: float):
    work = df.copy()

    numeric_cols = [c for c in REQUIRED_COLUMNS if c != "sku"] + [c for c in OPTIONAL_HISTORY_COLUMNS if c in work.columns]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["bias"] = work["actual"] - work["forecast"]
    work["abs_error"] = (work["actual"] - work["forecast"]).abs()
    work["bias_pct"] = pd.Series(safe_divide(work["bias"], work["forecast"].replace(0, np.nan)), index=work.index).fillna(0)
    work["wape"] = pd.Series(safe_divide(work["abs_error"], work["actual"].abs().replace(0, np.nan)), index=work.index).fillna(0)
    work["avg_daily_demand"] = (work[["forecast", "actual"]].mean(axis=1) / 30).replace(0, np.nan)
    work["coverage_days"] = pd.Series(
        safe_divide(work["inventory_on_hand"], work["avg_daily_demand"]), index=work.index
    ).replace([np.inf, -np.inf], np.nan).fillna(999)
    work["volatility_cv"] = compute_volatility(work)

    # --- MAPE FROM HISTORY ---
# --- MAPE FROM HISTORY ---
    if history_df is not None:
        required_cols = {"sku", "actual", "forecast"}

        if required_cols.issubset(history_df.columns):
            try:
                mape_df = (
                    history_df
                    .assign(ape=lambda x: abs(x["actual"] - x["forecast"]) / x["actual"].replace(0, 1))
                    .groupby("sku")["ape"]
                    .mean()
                    .reset_index()
                    .rename(columns={"ape": "mape"})
                )
                work = work.merge(mape_df, on="sku", how="left")

            except Exception:
                st.warning("Could not compute MAPE from uploaded history file. Skipping.")
                work["mape"] = None

        else:
            st.warning("Invalid forecast history file. Required columns: sku, actual, forecast.")
            work["mape"] = None

    else:
        work["mape"] = None

    # --- RAW RISK SIGNALS ---
    work["demand_risk_raw"] = (
        work["abs_error"]
        * work["volume"]
        * (1 + work["volatility_cv"])
        * (1 + work["mape"].clip(upper=0.5).fillna(0))
        
    )
    
    work["supply_risk_raw"] = (
        (1 - work["supplier_otif"].clip(0, 1))
        * work["lead_time_days"].clip(lower=0)
    )

    shortage_exposure = np.maximum(
        work["safety_stock"] - work["inventory_on_hand"],
        0
    )

    excess_exposure = np.maximum(
        work["inventory_on_hand"] - (work["safety_stock"] * 2),
        0
    )

    work["inventory_risk_raw"] = shortage_exposure + (0.25 * excess_exposure)

    # --- NORMALISED RISK SCORES ---
    work["demand_risk_score"] = min_max_scale(work["demand_risk_raw"])
    work["supply_risk_score"] = min_max_scale(work["supply_risk_raw"])
    work["inventory_risk_score"] = min_max_scale(work["inventory_risk_raw"])




    composite = (
        demand_weight * work["demand_risk_score"]
        + supply_weight * work["supply_risk_score"]
        + inventory_weight * work["inventory_risk_score"]
    )

    work["business_risk"] = composite * work["margin"].clip(lower=0) * work["volume"].clip(lower=0)
    work["business_risk"] = work["business_risk"].fillna(0)

    return work


def assign_action(df: pd.DataFrame, thresholds: RiskThresholds) -> pd.DataFrame:
    work = df.copy()
    high_risk_cutoff = work["business_risk"].quantile(thresholds.high_risk_quantile)

    conditions = [
        (work["business_risk"] >= high_risk_cutoff) & (work["coverage_days"] <= thresholds.low_coverage_days),
        (work["inventory_on_hand"] > work["safety_stock"] * 2)
        & (work["bias_pct"] <= thresholds.falling_demand_bias_pct),
        (work["supplier_otif"] < thresholds.poor_otif) | (work["lead_time_days"] > work["lead_time_days"].median()),
    ]
    actions = ["EXPEDITE", "REDUCE INVENTORY", "SUPPLY ISSUE"]
    work["recommended_action"] = np.select(conditions, actions, default="MONITOR")

    work["risk_drivers"] = (
        "Error " + work["abs_error"].round(0).astype(int).astype(str)
        + " · Coverage " + work["coverage_days"].round(1).astype(str)
        + "d · LT " + work["lead_time_days"].round(0).astype(int).astype(str)
        + "d · OTIF " + (work["supplier_otif"] * 100).round(0).astype(int).astype(str)
        + "%"
    )
    return work.sort_values("business_risk", ascending=False)


def build_template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku": ["SKU-1001", "SKU-1002", "SKU-1003", "SKU-1004", "SKU-1005"],
            "forecast": [1200, 800, 150, 600, 400],
            "actual": [1450, 700, 240, 550, 250],
            "volume": [1450, 700, 240, 550, 250],
            "margin": [3.2, 5.0, 12.0, 4.5, 8.5],
            "inventory_on_hand": [300, 1400, 500, 120, 950],
            "safety_stock": [450, 500, 120, 300, 400],
            "lead_time_days": [28, 14, 45, 21, 35],
            "supplier_otif": [0.82, 0.96, 0.74, 0.89, 0.78],
            "hist_1": [1100, 780, 130, 610, 430],
            "hist_2": [1180, 810, 160, 590, 410],
            "hist_3": [1250, 790, 200, 605, 390],
        }
    )


def style_actions(val):
    bg = ACTION_COLOR.get(val, "#9E9E9E")
    return f"background-color: {bg}20; color: {bg}; font-weight: 700;"


def style_risk(val):
    if val >= high:
        return "background-color: #f8d7da"
    elif val >= mid:
        return "background-color: #fff3cd"
    else:
        return ""
 


st.markdown('<div class="main-title">Planning Risk Prioritiser</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A decision-focused tool to prioritize SKU-level actions based on risk.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Risk Model")
    st.caption("Configure risk model behaviour.")
    demand_weight = st.slider("Demand risk", 0.0, 1.0, 0.5, 0.05)
    supply_weight = st.slider("Supply risk", 0.0, 1.0, 0.3, 0.05)
    inventory_weight = st.slider("Inventory risk", 0.0, 1.0, 0.2, 0.05)
    total = demand_weight + supply_weight + inventory_weight
    if total == 0:
        st.error("At least one risk driver must be greater than zero.")
    else:
        demand_weight /= total
        supply_weight /= total
        inventory_weight /= total

    st.header("Thresholds")
    low_coverage_days = st.number_input("Low coverage days", min_value=1.0, value=14.0)
    poor_otif = st.number_input("Poor OTIF threshold", min_value=0.0, max_value=1.0, value=0.85, step=0.01)

st.subheader("Upload data")
uploaded = st.file_uploader(
    "Current SKU planning data (required)",
    type=["csv"],
    key="main_upload"
)

history_uploaded = st.file_uploader(
    "Forecast vs actual history (optional)",
    type=["csv"],
    key="history_upload"
)

if uploaded is not None:
    st.session_state["uploaded_file"] = uploaded

if history_uploaded is not None:
    st.session_state["history_file"] = history_uploaded

margin_uploaded = st.file_uploader(
    "Upload margin data (optional)",
    type=["csv"],
    key="margin_upload"
)

supplier_uploaded = st.file_uploader(
    "Upload supplier data (optional)",
    type=["csv"],
    key="supplier_upload"
)

inventory_uploaded = st.file_uploader(
    "Upload inventory data (optional)",
    type=["csv"],
    key="inventory_upload"
)

leadtime_uploaded = st.file_uploader(
    "Upload lead time data (optional)",
    type=["csv"],
    key="leadtime upload"
)

template_df = build_template()
template_csv = template_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV template",
    data=template_csv,
    file_name="planning_risk_template.csv",
    mime="text/csv",
)

if "uploaded_file" not in st.session_state or st.session_state["uploaded_file"] is None:
    st.info("No core SKU file uploaded. Showing sample data. Upload your core file to run your scenario.")
    df = template_df.copy()
else:
    try:
        temp_df = pd.read_csv(
            st.session_state["uploaded_file"],
            sep=None,
            engine="python"
        )
        core_required = ["sku", "forecast", "actual", "volume"]
        missing_cols = [col for col in core_required if col not in temp_df.columns]

        if missing_cols:
            st.error("Core SKU file missing required columns: " + ", ".join(missing_cols) + ". Showing sample data instead."
            )
            df = template_df.copy()
        else:
            df = temp_df.copy()

    except Exception:
        st.warning("Core SKU file could not be read. Showing sample data instead.")
        df = template_df.copy()

margin_df = None

if margin_uploaded is not None:
    margin_df = pd.read_csv(margin_uploaded, sep=None, engine="python")

if margin_df is not None:
    required_cols = ["sku", "margin"]

    if not all(col in margin_df.columns for col in required_cols):
        st.error("Margin file must contain columns: 'sku' and 'margin'")
    else:
        df = df.merge(
            margin_df[["sku", "margin"]],
            on="sku",
            how="left",
            suffixes=("", "_uploaded")
        )
margin_df = None

if margin_uploaded is not None:
    try:
        margin_df = pd.read_csv(margin_uploaded, sep=None, engine="python")
    except Exception:
        st.warning("Could not read margin file. Skipping.")
        margin_df = None

if margin_df is not None:
    required_cols = ["sku", "margin"]

    if not all(col in margin_df.columns for col in required_cols):
        st.error("Margin file must contain columns: 'sku' and 'margin'")
    else:
        df = df.merge(
            margin_df[["sku", "margin"]],
            on="sku",
            how="left",
            suffixes=("", "_uploaded")
        )

        # ✅ Count BEFORE dropping
        matched = 0
        if "margin_uploaded" in df.columns:
            matched = df["margin_uploaded"].notna().sum()

        total = len(df)

        # ✅ Apply uploaded values
        if "margin_uploaded" in df.columns:
            df["margin"] = df["margin_uploaded"].combine_first(df["margin"])
            df = df.drop(columns=["margin_uploaded"])

        # ✅ Split logic
        if matched == 0:
            if len(margin_df) > 0:
                st.warning("Margin file uploaded, but no matching SKUs were found. Check SKU alignment.")
            else:
                st.error("Margin file appears empty or unreadable.")
        else:
            st.success(f"Margin data merged for {matched:,}/{total:,} SKUs.")

        st.caption("Preview of uploaded margin data")
        st.dataframe(margin_df.head())

supplier_df = None

if supplier_uploaded is not None:
    supplier_df = pd.read_csv(supplier_uploaded, sep=None, engine="python")

if supplier_df is not None:
    required_cols = ["sku", "supplier_otif"]

    if not all(col in supplier_df.columns for col in required_cols):
        st.error("Supplier file must contain columns: 'sku' and 'supplier_otif'")
    else:
        df = df.merge(
            supplier_df[["sku", "supplier_otif"]],
            on="sku",
            how="left",
            suffixes=("", "_uploaded")
        )

        # ✅ COUNT BEFORE modifying
        if "supplier_otif_uploaded" in df.columns:
            matched = df["supplier_otif_uploaded"].notna().sum()
        else:
            matched = 0

        total = len(df)

        if matched == 0:
            if len (supplier_df) > 0:
                st.warning("Supplier file uploaded, but no matching SKUs were found")
            else:
                st.error("Supplier file appears empty or unreadable")
        else:
            st.success(f"Supplier data merged for {matched:,}/{total:,} SKUs.")

        # ✅ THEN combine
        if "supplier_otif_uploaded" in df.columns:
            df["supplier_otif"] = df["supplier_otif_uploaded"].combine_first(df["supplier_otif"])
            df = df.drop(columns=["supplier_otif_uploaded"])

        st.caption("Preview of uploaded supplier data")
        st.dataframe(supplier_df.head())

inventory_df = None

if inventory_uploaded is not None:
    inventory_df = pd.read_csv(inventory_uploaded, sep=None, engine="python")

if inventory_df is not None:
    required_cols = ["sku", "inventory_on_hand", "safety_stock"]

    if not all(col in inventory_df.columns for col in required_cols):
        st.error("Inventory file must contain columns: 'sku', 'inventory_on_hand', and 'safety_stock'")
    else:
        df = df.merge(
            inventory_df[["sku", "inventory_on_hand", "safety_stock"]],
            on="sku",
            how="left",
            suffixes=("", "_uploaded")
        )

        # ✅ Calculate matches BEFORE dropping columns
        matched = 0
        if "inventory_on_hand_uploaded" in df.columns:
            matched = df["inventory_on_hand_uploaded"].notna().sum()

        total = len(df)

        # ✅ Apply uploaded values
        for col in ["inventory_on_hand", "safety_stock"]:
            uploaded_col = f"{col}_uploaded"
            if uploaded_col in df.columns:
                df[col] = df[uploaded_col].combine_first(df[col])
                df = df.drop(columns=[uploaded_col])

        # ✅ Messaging
        if matched == 0:
            if len (inventory_df) > 0:
                st.warning("Inventory file uploaded, but no matching SKUs were found. Check SKU alignment.")
            else:
                st.error("Inventory file appears empty or unreadable.")
        else:
            st.success(f"Inventory data merged for {matched:,}/{total:,} SKUs.")

        st.caption("Preview of uploaded inventory data")
        st.dataframe(inventory_df.head())

leadtime_df = None

if leadtime_uploaded is not None:
    leadtime_df = pd.read_csv(leadtime_uploaded, sep=None, engine="python")

if leadtime_df is not None:
    required_cols = ["sku", "lead_time_days"]

    if not all(col in leadtime_df.columns for col in required_cols):
        st.error("Lead time file must contain columns: 'sku' and 'lead_time_days'")
    else:
        df = df.merge(
            leadtime_df[["sku", "lead_time_days"]],
            on="sku",
            how="left",
            suffixes=("", "_uploaded")
        )

        # Count uploaded matches BEFORE dropping the uploaded column
        matched = 0
        if "lead_time_days_uploaded" in df.columns:
            matched = df["lead_time_days_uploaded"].notna().sum()

        total = len(df)

        # Apply uploaded values
        if "lead_time_days_uploaded" in df.columns:
            df["lead_time_days"] = df["lead_time_days_uploaded"].combine_first(df["lead_time_days"])
            df = df.drop(columns=["lead_time_days_uploaded"])

        # Messaging
        if matched == 0:
            if len(leadtime_df) > 0:
                st.warning("Lead time file uploaded, but no matching SKUs were found. Check file content for SKU alignment.")
            else:
                st.success(f"Lead time data merged for {matched:,}/{total:,} SKUs.")

        st.caption("Preview of uploaded lead time data")
        st.dataframe(leadtime_df.head())

history_df = None

if "history_file" in st.session_state and st.session_state["history_file"] is not None:
    try:
        history_df = pd.read_csv(
            st.session_state["history_file"],
            sep=None,
            engine="python"
        )
    except Exception:
        st.warning("File skipped: invalid format or missing required columns.")
        history_df = None

thresholds = RiskThresholds(low_coverage_days=low_coverage_days, poor_otif=poor_otif)
metrics_df = compute_metrics(df, demand_weight, supply_weight, inventory_weight)
result_df = assign_action(metrics_df, thresholds)

max_top_n = max(1, min(50, len(result_df)))
default_top_n = min(10, max_top_n)
if max_top_n == 1:
    top_n = 1
    st.caption("Only 1 SKU available in the current filtered dataset.")
elif max_top_n <= 5:
    top_n = st.slider("Priority SKUs to display", 1, max_top_n, default_top_n)
else:
    top_n = st.slider("Priority SKUs to display", 5, max_top_n, default_top_n)


total_skus = len(result_df)

# High risk definition (top 25%)
high_risk_cutoff = result_df["business_risk"].quantile(0.75)
high_risk_count = int((result_df["business_risk"] >= high_risk_cutoff).sum())

# Total risk
total_risk = float(result_df["business_risk"].sum())

# Top 3 concentration (CORRECT)
risk_top3 = float(
    result_df
    .sort_values("business_risk", ascending=False)
    .head(3)["business_risk"]
    .sum()
)

risk_concentration = 0 if total_risk == 0 else risk_top3 / total_risk

summary_box = st.container(border=True)

summary_box.markdown("### Summary")
summary_box.caption("At-a-glance view of risk scale and concentration.")

col1, col2, col3, col4 = summary_box.columns(4)

col1.metric(
"SKUs",
total_skus,
)

col2.metric(
"High-Risk",
high_risk_count,
delta=f"{high_risk_count/total_skus:.0%} of total",
delta_color="inverse" # 🔥 makes higher = worse (red)
)

col3.metric(
"Risk Exposure",
f"€{total_risk:,.0f}",
)

col4.metric(
"Top 3 Share",
f"{risk_concentration:.1%}",
delta="concentration",
delta_color="inverse" # 🔥 high concentration = bad → red
)



show_cols = [
    "sku",
    "recommended_action",
    "business_risk",
    "risk_drivers",
    "bias",
    "abs_error",
    "coverage_days",
    "lead_time_days",
    "supplier_otif",
]

display_df = result_df[show_cols].head(top_n).copy()
display_df["Recommended Action"] = display_df["recommended_action"].map(format_action)
display_df["Risk (€)"] = display_df["business_risk"].round(0)
display_df["Coverage (days)"] = display_df["coverage_days"].round(1)
display_df["Supplier OTIF (%)"] = (display_df["supplier_otif"] * 100).round(1)
display_df["Bias"] = display_df["bias"].round(0)
display_df["Abs Error"] = display_df["abs_error"].round(0)
display_df["Lead Time (days)"] = display_df["lead_time_days"].round(0)
display_df["SKU"] = display_df["sku"]
display_df["Risk Drivers"] = display_df["risk_drivers"]

final_table = display_df[[
    "SKU",
    "Recommended Action",
    "Risk (€)",
    "Risk Drivers",
    "Bias",
    "Abs Error",
    "Coverage (days)",
    "Lead Time (days)",
    "Supplier OTIF (%)",
]].copy()

# --- KPI CALCULATIONS ---
total_skus = len(final_table)

high_risk_threshold = final_table["Risk (€)"].quantile(0.75)
high_risk_skus = final_table[final_table["Risk (€)"] >= high_risk_threshold]

total_risk = final_table["Risk (€)"].sum()
high = final_table["Risk (€)"].quantile(0.75)
mid = final_table["Risk (€)"].quantile(0.40)
top3_risk = (
    final_table.sort_values("Risk (€)", ascending=False)
    .head(3)["Risk (€)"]
    .sum()
)

top3_share = (top3_risk / total_risk * 100) if total_risk > 0 else 0


# --- KPI DISPLAY ---
col1, col2, col3, col4 = st.columns(4)


st.markdown("---")
st.subheader("Priority SKUs")
st.markdown("***SKUs ranked by highest risk exposure***")

final_table = final_table.sort_values("Risk (€)", ascending=False)

st.dataframe(
    final_table.style
    .apply(
        lambda s: [style_actions(v) if s.name == "Recommended Action" else "" for v in s],
        axis=0,
    )
    .apply(
        lambda s: [style_risk(v) if s.name == "Risk (€)" else "" for v in s],
        axis=0,
    )
    .format({"Risk (€)": "{:,.0f}"}),
    use_container_width=True,
    hide_index=True,
)
st.subheader("Risk Exposure")
st.markdown('<div class="section-note">Highest-risk SKUs first. Useful for screenshots and quick review.</div>', unsafe_allow_html=True)
chart_df = (
    result_df[["sku", "business_risk"]]
    .head(top_n)
    .sort_values("business_risk", ascending=False)
    .rename(columns={"business_risk": "Risk (€)"})
)

st.bar_chart(chart_df, x="sku", y="Risk (€)", horizontal=True)

st.subheader("SKU drill-down")

selected_sku = st.selectbox("Select SKU", result_df["sku"].tolist())
selected = result_df[result_df["sku"] == selected_sku].iloc[0]

# --- COMPUTE HISTORY + MAPE FIRST ---
mape = None
sku_history = None

if history_df is not None and {"sku", "actual", "forecast"}.issubset(history_df.columns):
    sku_history = history_df[history_df["sku"] == selected_sku]

    if not sku_history.empty:
        sku_history["ape"] = sku_history.apply(
            lambda r: abs(r["actual"] - r["forecast"]) / r["actual"] if r["actual"] != 0 else 0,
            axis=1
        )
        mape = sku_history["ape"].mean()

elif history_df is not None:
    st.warning("Invalid forecast history file. Required columns: sku, actual, forecast.")

else:
    sku_history = None

# --- DISPLAY METRICS ---
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Risk (€)", f"€{selected['business_risk']:,.0f}")
k2.metric("Recommended Action", selected["recommended_action"])
k3.metric("Coverage Days", f"{selected['coverage_days']:.1f}")
k4.metric("Supplier OTIF", f"{selected['supplier_otif'] * 100:.1f}%")

if mape is not None:
    k5.metric("MAPE", f"{mape * 100:.1f}%")
else:
    k5.metric("MAPE", "N/A")

# --- FORECAST VS ACTUAL CHART ---
if sku_history is not None and not sku_history.empty:
    sku_history = sku_history.sort_values("date")

    st.markdown("#### Forecast vs Actual")

    sku_history["error"] = sku_history["actual"] - sku_history["forecast"]

    st.line_chart(
        sku_history.set_index("date")[["forecast", "actual", "error"]]
    )

else:
    st.warning("No historical data available for this SKU")

# --- RISK DRIVERS ---
reasons = explain_sku_risk(selected)

st.markdown("### Risk drivers")

action = selected.get("recommended_action", "").upper()

clean_reasons = " • ".join(reasons)

if action == "MONITOR":
    st.success("✔ No risk detected")

elif action == "EXPEDITE":
    st.error(f"🚨 {clean_reasons}")

elif action == "SUPPLY ISSUE":
    st.warning(f"⚠ {clean_reasons}")

elif action == "REDUCE INVENTORY":
    st.info(f"📦 {clean_reasons}")

else:
    st.warning(f"⚠ {clean_reasons}")

explain_df = pd.DataFrame(
    {
        "Metric": [
            "Bias",
            "Absolute Error",
            "Demand Volatility CV",
            "Lead Time (days)",
            "Inventory on Hand",
            "Safety Stock",
            "Margin",
        ],
        "Value": [
            round(float(selected["bias"]), 2),
            round(float(selected["abs_error"]), 2),
            round(float(selected["volatility_cv"]), 3),
            round(float(selected["lead_time_days"]), 1),
            round(float(selected["inventory_on_hand"]), 1),
            round(float(selected["safety_stock"]), 1),
            round(float(selected["margin"]), 2),
        ],
    }
)
st.dataframe(explain_df, use_container_width=True, hide_index=True)

st.markdown(f"""
---
**How the model works**

- **Demand risk:** forecast error × volume (adjusted for volatility)  
- **Supply risk:** low OTIF and long lead times increase risk  
- **Inventory risk:**  
  • Low coverage → shortage risk  
  • High stock → excess inventory risk  
- **Business impact:** risk weighted by margin and volume  
""")
