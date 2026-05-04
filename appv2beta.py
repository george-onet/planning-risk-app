# =============================================================================
# Planning Risk Prioritiser - v3 (WMAPE)
# =============================================================================
# CHANGE FROM v2:
#   MAPE replaced with WMAPE throughout:
#   WMAPE = sum(|actual - forecast|) / sum(actual) x 100
#   - Volume-weighted: high-volume SKU misses dominate the score
#   - No division-by-zero risk on low-volume periods
#   - Industry standard for pharma and chemical supply chain
#
# All other logic, fixes, and features from v2 are preserved.
# =============================================================================

import io
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# CONSTANTS
# =============================================================================

LOG_FILE = "usage_log.csv"

REQUIRED_COLUMNS = [
    "sku", "forecast", "actual", "volume",
    "margin", "inventory_on_hand", "safety_stock",
    "lead_time_days", "supplier_otif",
]
OPTIONAL_HISTORY_COLUMNS = ["hist_1", "hist_2", "hist_3"]

ACTION_ICON = {
    "EXPEDITE":         "🔴",
    "SUPPLY ISSUE":     "🟠",
    "OPTIMISE INVENTORY": "🟡",
    "REVIEW CONTRACT":  "🔵",
    "MONITOR":          "🟢",
    "REVIEW FORECAST": "🟣",

}
ACTION_COLOR = {
    "EXPEDITE":         "#C62828",
    "SUPPLY ISSUE":     "#EF6C00",
    "OPTIMISE INVENTORY": "#F9A825",
    "REVIEW CONTRACT":  "#1565C0",
    "MONITOR":          "#2E7D32",
    "REVIEW FORECAST": "#6A1B9A",
}

ACTION_STATUS_OPTIONS = ["Open", "In Progress", "Resolved"]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RiskThresholds:
    high_risk_quantile:      float = 0.75
    low_coverage_days:       float = 14.0
    high_coverage_days:      float = 45.0
    falling_demand_bias_pct: float = -0.10
    poor_otif:               float = 0.85


# =============================================================================
# HELPERS
# =============================================================================

def normalize_sku(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.strip().str.upper()
        .str.replace("-", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.replace(" ", "", regex=False)
    )


def clean_number(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.strip()
        .str.replace("%",    "", regex=False)
        .str.replace("€",   "", regex=False)
        .str.replace("days","", regex=False)
        .str.replace("DAYS","", regex=False)
        .str.replace(",",   ".", regex=False)
        .str.replace(r"(\d)\s*d$", r"\1", regex=True)  # strips "45d" → "45"
    )


def safe_divide(a, b):
    b = np.where(np.asarray(b) == 0, np.nan, b)
    return np.asarray(a) / b


def min_max_scale(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.ones(len(s)) * 0.5, index=s.index)
    return (s - lo) / (hi - lo)


def format_action(action: str) -> str:
    return f"{ACTION_ICON.get(action, '⚪')} {action}"


def log_event(event_name: str) -> None:
    row = pd.DataFrame([{"timestamp": datetime.now(), "event": event_name}])
    row.to_csv(
        LOG_FILE,
        mode="a",
        header=not os.path.exists(LOG_FILE),
        index=False,
    )


def _read_file(f) -> pd.DataFrame:
    if f.name.lower().endswith(".xlsx"):
        return pd.read_excel(f)
    return pd.read_csv(f, sep=None, engine="python")


def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str).str.strip().str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("\ufeff", "", regex=False)
    )
    return df


RENAME_MAIN = {
    "sku": "sku", "product_code": "sku", "productcode": "sku",
    "product": "sku", "material": "sku", "item": "sku",
    "forecast": "forecast", "forecast_volume": "forecast", "forecast_qty": "forecast",
    "actual": "actual", "actual_volume": "actual", "sales": "actual", "demand": "actual",
    "volume": "volume", "qty": "volume", "quantity": "volume",
    "inventory_on_hand": "inventory_on_hand", "stock_on_hand": "inventory_on_hand",
    "on_hand": "inventory_on_hand",
    "safety_stock": "safety_stock", "ss": "safety_stock",
    "lead_time": "lead_time_days", "leadtime": "lead_time_days",
    "lead_time_days": "lead_time_days", "lt": "lead_time_days",
    "supplier_otif": "supplier_otif", "otif": "supplier_otif",
    "vendor_otif": "supplier_otif",
    "margin": "margin", "gross_margin": "margin",
    "margin_pct": "margin", "margin_%": "margin",
}


# =============================================================================
# GENERIC OPTIONAL FILE LOADER
# =============================================================================

def _load_optional(
    file,
    rename_map: dict,
    required_cols: List[str],
    label: str,
) -> Optional[pd.DataFrame]:
    try:
        raw = _read_file(file)
        raw = _std_cols(raw)
        raw.rename(columns=rename_map, inplace=True)
        missing = [c for c in required_cols if c not in raw.columns]
        if missing:
            st.error(f"{label}: missing columns — {', '.join(missing)}")
            return None
        if "sku" in raw.columns:
            raw["sku"] = normalize_sku(raw["sku"])
        st.success(f"✓ {label} loaded")
        return raw
    except Exception as e:
        st.error(f"✕ {label} failed to load: {e}")
        return None


# =============================================================================
# WMAPE CALCULATION
# Formula: sum(|actual - forecast|) / sum(actual) x 100
# - Weights errors by actual volume — high-volume misses dominate
# - No division-by-zero risk on individual periods
# - Industry standard for pharma and chemical supply chain
# =============================================================================

def compute_wmape_sku(actual: pd.Series, forecast: pd.Series) -> Optional[float]:
    """
    WMAPE at SKU level across all periods.
    Returns None if actual demand is zero across all periods.
    """
    actual   = pd.to_numeric(actual,   errors="coerce").fillna(0)
    forecast = pd.to_numeric(forecast, errors="coerce").fillna(0)
    total_actual = actual.abs().sum()
    if total_actual == 0:
        return None
    return (actual - forecast).abs().sum() / total_actual * 100


def compute_wmape_portfolio(history_df: pd.DataFrame) -> Optional[float]:
    """
    WMAPE at portfolio level across all SKUs and all periods.
    Single number representing overall forecast accuracy.
    """
    if history_df is None:
        return None
    actual   = pd.to_numeric(history_df["actual"],   errors="coerce").fillna(0)
    forecast = pd.to_numeric(history_df["forecast"], errors="coerce").fillna(0)
    total_actual = actual.abs().sum()
    if total_actual == 0:
        return None
    return (actual - forecast).abs().sum() / total_actual * 100

# =============================================================================
# LEAD TIME TREND
# =============================================================================

def compute_leadtime_trend(lt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates lead time trend per SKU from historical data.
    Requires columns: sku, date, lead_time_days
    Returns a DataFrame with sku and leadtime_trend columns.
    """
    if lt_df is None:
        return pd.DataFrame(columns=["sku", "leadtime_trend"])

    if not {"sku", "date", "lead_time_days"}.issubset(lt_df.columns):
        return pd.DataFrame(columns=["sku", "leadtime_trend"])

    results = []

    for sku, group in lt_df.groupby("sku"):
        group = group.sort_values("date").dropna(subset=["lead_time_days"])

        if len(group) < 2:
            results.append({"sku": sku, "leadtime_trend": "🟢 Stable"})
            continue

        values = group["lead_time_days"].values.astype(float)
        x = np.arange(len(values), dtype=float)

        # Slope relative to mean — same logic as demand trend
        slope = np.polyfit(x, values, 1)[0]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=0)

        # Relative slope: how much is it changing as % of average
        rel_slope = slope / mean_val if mean_val > 0 else 0

        # Variability: std as % of mean (coefficient of variation)
        cv = std_val / mean_val if mean_val > 0 else 0

        high_variability = cv > 0.15  # >15% variation = unstable

        if rel_slope > 0.10:          # growing >10% per period
            if high_variability:
                trend = "🔴 Deteriorating"
            else:
                trend = "🟠 Lengthening"
        elif rel_slope < -0.10:       # shrinking >10% per period
            trend = "🟢 Improving"
        else:
            if high_variability:
                trend = "🟡 Unstable"
            else:
                trend = "🟢 Stable"

        results.append({"sku": sku, "leadtime_trend": trend})

    return pd.DataFrame(results)

# =============================================================================
# ANALYTICS — VOLATILITY & DEMAND TREND
# =============================================================================

def compute_volatility(df: pd.DataFrame) -> pd.Series:
    hist_cols = [c for c in OPTIONAL_HISTORY_COLUMNS if c in df.columns]
    if len(hist_cols) >= 2:
        hist   = df[hist_cols].astype(float)
        mean_h = hist.mean(axis=1).replace(0, np.nan)
        return (hist.std(axis=1, ddof=0) / mean_h).fillna(0)
    return pd.Series(np.zeros(len(df)), index=df.index)


def compute_demand_trend(df: pd.DataFrame) -> pd.Series:
    """3-period slope from hist columns. Returns up / stable / down per SKU."""
    hist_cols = [c for c in OPTIONAL_HISTORY_COLUMNS if c in df.columns]
    if len(hist_cols) < 2:
        return pd.Series(["→"] * len(df), index=df.index)

    hist = df[hist_cols].astype(float)
    x    = np.arange(hist.shape[1], dtype=float)

    def _direction(row):
        y = row.values.astype(float)
        if np.isnan(y).all():
            return "→"
        slope  = np.polyfit(x, y, 1)[0]
        mean_y = np.nanmean(y)
        if mean_y == 0:
            return "→"
        rel = slope / mean_y
        if rel >  0.05:
            return "↑"
        if rel < -0.05:
            return "↓"
        return "→"

    return hist.apply(_direction, axis=1)

def compute_demand_trend_history(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates demand trend per SKU from actual demand history.
    Requires columns: sku, date, actual
    Returns DataFrame with sku and demand_trend columns.
    """
    if history_df is None:
        return pd.DataFrame(columns=["sku", "demand_trend"])

    if not {"sku", "date", "actual"}.issubset(history_df.columns):
        return pd.DataFrame(columns=["sku", "demand_trend"])

    results = []

    for sku, group in history_df.groupby("sku"):
        group = group.sort_values("date").dropna(subset=["actual"])

        if len(group) < 2:
            results.append({"sku": sku, "demand_trend": "→"})
            continue

        values = group["actual"].values.astype(float)
        x      = np.arange(len(values), dtype=float)

        slope  = np.polyfit(x, values, 1)[0]
        mean_y = np.mean(values)

        if mean_y == 0:
            results.append({"sku": sku, "demand_trend": "→"})
            continue

        rel = slope / mean_y

        if rel > 0.05:
            trend = "↑"
        elif rel < -0.05:
            trend = "↓"
        else:
            trend = "→"

        results.append({"sku": sku, "demand_trend": trend})

    return pd.DataFrame(results)

# =============================================================================
# CORE METRIC COMPUTATION
# WMAPE replaces MAPE in risk scoring
# =============================================================================

def compute_metrics(
    df: pd.DataFrame,
    demand_weight:         float,
    supply_weight:         float,
    inventory_weight:      float,
    history_df:            Optional[pd.DataFrame] = None,
    demand_change_pct:     float = 0,
    lead_time_change_days: float = 0,
    otif_change_pct:       float = 0,
    lt_df=None
) -> pd.DataFrame:

    work = df.copy()

    numeric_cols = (
        [c for c in REQUIRED_COLUMNS if c != "sku"]
        + [c for c in OPTIONAL_HISTORY_COLUMNS if c in work.columns]
    )
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # --- Scenario adjustments ---
    work["forecast"]       = work["forecast"]       * (1 + demand_change_pct    / 100)
    work["lead_time_days"] = work["lead_time_days"] + lead_time_change_days
    work["supplier_otif"]  = (work["supplier_otif"] * (1 + otif_change_pct / 100)).clip(0, 1)

    # --- Derived fields ---
    work["bias"]      = work["actual"] - work["forecast"]
    work["abs_error"] = work["bias"].abs()
    work["bias_pct"]  = pd.Series(
        safe_divide(work["bias"], work["forecast"].replace(0, np.nan)),
        index=work.index,
    ).fillna(0)

    work["avg_daily_demand"] = (work[["forecast", "actual"]].mean(axis=1) / 30).replace(0, np.nan)
    work["coverage_days"] = pd.Series(
        safe_divide(work["inventory_on_hand"], work["avg_daily_demand"]),
        index=work.index,
    ).replace([np.inf, -np.inf], np.nan).fillna(999)

    work["volatility_cv"]  = compute_volatility(work)
    if history_df is not None and {"sku", "date", "actual"}.issubset(history_df.columns):
        demand_trend_df = compute_demand_trend_history(history_df)
        work = work.merge(demand_trend_df, on="sku", how="left")
        work["demand_trend"] = work["demand_trend"].fillna("→")
    else:
        work["demand_trend"] = compute_demand_trend(work)

    # --- Lead time trend - only if history file has a date column

    if lt_df is not None and "date" in lt_df.columns:
        lt_trend_df = compute_leadtime_trend(lt_df)
        work = work.merge(lt_trend_df, on="sku", how="left")
        work["leadtime_trend"] = work["leadtime_trend"].fillna("🟢 Stable")
    else:
        work["leadtime_trend"] = "🟢 Stable"

    excess_units = np.maximum(work["inventory_on_hand"] - work["safety_stock"] * 2, 0)
    work["excess_stock_value"] = excess_units * work["margin"].clip(lower=0)

    # Zero out excess stock when coverage is below lead time — not truly excess if stockout is incoming
    work.loc[work["coverage_days"] < work["lead_time_days"], "excess_stock_value"] = 0


    # --- WMAPE from history (replaces MAPE) ---
    # At SKU level: sum(|actual - forecast|) / sum(actual) x 100
    if history_df is not None and {"sku", "actual", "forecast"}.issubset(history_df.columns):
        try:
            wmape_df = (
                history_df
                .groupby("sku")
                .apply(
                    lambda g: compute_wmape_sku(g["actual"], g["forecast"])
                )
                .reset_index()
                .rename(columns={0: "wmape"})
            )
            work = work.merge(wmape_df, on="sku", how="left")
        except Exception:
            work["wmape"] = np.nan
    else:
        work["wmape"] = np.nan

    # --- Raw risk signals ---
    # WMAPE now drives demand risk accuracy component (replaces MAPE)
    work["demand_risk_raw"] = (
        work["abs_error"]
        * work["volume"]
        * (1 + work["volatility_cv"])
        * (1 + (work["wmape"] / 100).clip(upper=0.5).fillna(0))
    )
    work["supply_risk_raw"] = (
        (1 - work["supplier_otif"].clip(0, 1))
        * work["lead_time_days"].clip(lower=0)
    )

    shortage_exposure = np.maximum(work["safety_stock"] - work["inventory_on_hand"], 0)
    excess_exposure   = np.maximum(work["inventory_on_hand"] - work["safety_stock"] * 2, 0)
    work["inventory_risk_raw"] = shortage_exposure + 0.25 * excess_exposure

    # --- Normalised scores ---
    work["demand_risk_score"]    = min_max_scale(work["demand_risk_raw"])
    work["supply_risk_score"]    = min_max_scale(work["supply_risk_raw"])
    work["inventory_risk_score"] = min_max_scale(work["inventory_risk_raw"])

    composite = (
        demand_weight      * work["demand_risk_score"]
        + supply_weight    * work["supply_risk_score"]
        + inventory_weight * work["inventory_risk_score"]
    )

    # --- FINANCIAL EXPOSURE (S&OP / finance-owned) ---
    units_short = np.maximum(work["safety_stock"] - work["inventory_on_hand"], 0)

    stockout_days = np.maximum(work["lead_time_days"] - work["coverage_days"], 0)
    shortfall_units = stockout_days * work["avg_daily_demand"].fillna(0)

    work["revenue_at_risk"] = (
        np.maximum(units_short, shortfall_units)
        * work["margin"].abs().clip(lower=0.1)
    ).fillna(0)

    # --- BUSINESS RISK (combined — keeps ranking working) ---
    work["business_risk"] = (
        composite * work["margin"].abs().clip(lower=0.1) * work["volume"].clip(lower=0)
    ).fillna(0)

    return work


# =============================================================================
# ACTION ASSIGNMENT
# =============================================================================

def assign_action(
    df: pd.DataFrame,
    thresholds:      RiskThresholds,
    contract_active: bool = False,
    moq_waived:      bool = False,
) -> pd.DataFrame:

    work = df.copy()

    for col, default in {
        "call_off_volume": 0, "moq": 0,
        "supplier_locked": False, "strategic_customer": False,
    }.items():
        if col not in work.columns:
            work[col] = default

    conditions = [
        work["coverage_days"] <= 3,

        (work["inventory_on_hand"] < work["safety_stock"])
        | ((work["supplier_otif"] < thresholds.poor_otif) & (work["coverage_days"] < work["lead_time_days"]))        | (work["lead_time_days"] > 60)
        | (work["coverage_days"] < work["lead_time_days"]),

        (
            (contract_active)
            & (~moq_waived)
            & (work["call_off_volume"] > 0)
            & (work["coverage_days"] > 60)
        ),
        (
        ((work["coverage_days"] > 60) & (work["inventory_on_hand"] > work["safety_stock"] * 2))
            | ((work["inventory_on_hand"] > work["safety_stock"] * 3) & (work["excess_stock_value"] > 50_000))
        ),

        (work["wmape"] > 40)
        & (work["coverage_days"] > 3),
    ]

    actions = [
        "EXPEDITE",
        "SUPPLY ISSUE",
        "REVIEW CONTRACT",
        "OPTIMISE INVENTORY",
        "REVIEW FORECAST",
    ]

    work["recommended_action"] = np.select(conditions, actions, default="MONITOR")

    work["risk_drivers"] = (
        "Error "     + work["abs_error"].round(0).astype(int).astype(str)
        + " · Cov "  + work["coverage_days"].round(1).astype(str)  + "d"
        + " · LT "   + work["lead_time_days"].round(0).astype(int).astype(str) + "d"
        + " · OTIF " + (work["supplier_otif"] * 100).round(0).astype(int).astype(str) + "%"
    )

    return work.sort_values("business_risk", ascending=False)


# =============================================================================
# RISK DRIVER EXPLANATION
# =============================================================================

def explain_sku_risk(row: pd.Series) -> List[str]:
    drivers   = []
    bias      = row["actual"] - row["forecast"]
    inventory = row["inventory_on_hand"]
    safety    = row["safety_stock"]
    coverage  = row["coverage_days"]
    lead      = row["lead_time_days"]
    otif      = row["supplier_otif"]
    margin    = row["margin"]
    wmape      = row.get("wmape", None)
    excess_val = row.get("excess_stock_value", 0)

    if inventory < safety:
        drivers.append(("Inventory below safety stock", 1000))
    if coverage < 3:
        drivers.append(("Critical stock coverage — expedite required", 950))
    if coverage < lead:
        drivers.append(("Coverage below lead time — stockout before replenishment", 850))
    elif coverage < 14:
        drivers.append(("Low stock coverage", 700))
    if otif < 0.85:
        drivers.append(("Supplier reliability risk", 800))
    if lead > 30:
        drivers.append(("Long replenishment lead time", 600))
    if bias > 0 and inventory < safety:
        drivers.append(("Under-forecasting / stockout risk", abs(bias)))
    elif bias < 0 and inventory > safety * 1.5:
        drivers.append(("Over-forecasting / excess inventory risk", abs(bias)))
    if wmape is not None and not pd.isna(wmape) and wmape > 40:
        drivers.append(("Poor forecast accuracy — WMAPE above 40%", 750))
    if inventory > safety * 2 and excess_val > 25_000:
        if otif < 0.70:
            drivers.append(("Reduce inventory cautiously — supplier reliability is low", 660))
        else:
            drivers.append(("Excess inventory — working capital exposure", 650))
    if margin >= 10:
        drivers.append(("High margin exposure", 500))

    if not drivers:
        return ["Risk appears controlled"]

    return [d[0] for d in sorted(drivers, key=lambda x: x[1], reverse=True)[:3]]


# =============================================================================
# STYLE HELPERS
# =============================================================================

def style_actions(val: str) -> str:
    bg = ACTION_COLOR.get(val, "#9E9E9E")
    return f"background-color: {bg}20; color: {bg}; font-weight: 700;"


def make_style_risk(high: float, mid: float):
    def _style(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v >= high:
            return "background-color: #f8d7da; color: #111;"
        if v >= mid:
            return "background-color: #fff3cd; color: #111;"
        return ""
    return _style


# =============================================================================
# TEMPLATE & EXCEL EXPORT
# =============================================================================

def build_template() -> pd.DataFrame:
    return pd.DataFrame({
        "sku":               ["SKU-1001","SKU-1002","SKU-1003","SKU-1004","SKU-1005"],
        "forecast":          [1200, 800, 150, 600, 400],
        "actual":            [1450, 700, 240, 550, 250],
        "volume":            [1450, 700, 240, 550, 250],
        "margin":            [3.2,  5.0, 12.0,  4.5,  8.5],
        "inventory_on_hand": [300, 1400,  500,  120,  950],
        "safety_stock":      [450,  500,  120,  300,  400],
        "lead_time_days":    [28,   14,   45,   21,   35],
        "supplier_otif":     [0.82, 0.96, 0.74, 0.89, 0.78],
        "hist_1":            [1100, 780,  130,  610,  430],
        "hist_2":            [1180, 810,  160,  590,  410],
        "hist_3":            [1250, 790,  200,  605,  390],
    })


def build_excel_export(result_df: pd.DataFrame) -> bytes:
    col_map = {
        "sku":               "SKU",
        "recommended_action":"Action",
        "action_status":     "Status",
        "business_risk":     "Risk (EUR)",
        "demand_trend":      "Trend",
        "wmape":             "WMAPE (%)",
        "risk_drivers":      "Risk Drivers",
        "coverage_days":     "Coverage (days)",
        "lead_time_days":    "Lead Time (days)",
        "supplier_otif":     "OTIF (%)",
        "excess_stock_value":"Excess Stock (EUR)",
        "bias":              "Bias",
        "abs_error":         "Abs Error",
        "margin":            "Margin",
        "volume":            "Volume",
    }
    available = {k: v for k, v in col_map.items() if k in result_df.columns}
    export    = result_df[list(available.keys())].copy()
    export.columns = list(available.values())

    if "OTIF (%)" in export.columns:
        export["OTIF (%)"] = (export["OTIF (%)"] * 100).round(1)
    if "Risk (EUR)" in export.columns:
        export["Risk (EUR)"] = export["Risk (EUR)"].round(0)
    if "Excess Stock (EUR)" in export.columns:
        export["Excess Stock (EUR)"] = export["Excess Stock (EUR)"].round(0)
    if "WMAPE (%)" in export.columns:
        export["WMAPE (%)"] = export["WMAPE (%)"].round(1)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        export.to_excel(writer, index=False, sheet_name="Priority Actions")
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# PAGE CONFIG & STYLES
# =============================================================================

st.set_page_config(page_title="PlanSignal", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 3rem; padding-bottom: 2rem; }
.main-title      { font-size: 2.4rem; font-weight: 800; margin-bottom: 0.15rem; letter-spacing: -0.5px; }
.subtitle        { color: #5f6368; margin-bottom: 1.25rem; font-size: 1.05rem; }
.section-note    { color: #5f6368; font-size: 0.92rem; margin-bottom: 0.6rem; margin-top: -0.2rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if "visited" not in st.session_state:
    log_event("app_opened")
    st.session_state["visited"] = True

if "action_status" not in st.session_state:
    st.session_state["action_status"] = {}


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("Risk Weights")
    demand_weight    = st.slider("Demand risk",    0.0, 1.0, 0.5, 0.05)
    supply_weight    = st.slider("Supply risk",    0.0, 1.0, 0.3, 0.05)
    inventory_weight = st.slider("Inventory risk", 0.0, 1.0, 0.2, 0.05)

    _total = demand_weight + supply_weight + inventory_weight
    if _total == 0:
        st.error("At least one weight must be > 0.")
        st.stop()
    demand_weight    /= _total
    supply_weight    /= _total
    inventory_weight /= _total

    st.caption(
        f"Normalised: Demand {demand_weight:.0%} · "
        f"Supply {supply_weight:.0%} · "
        f"Inventory {inventory_weight:.0%}"
    )

    st.header("Scenario Pack")
    demand_change_pct     = st.slider("Demand change (%)",       -30, 30, 0, 5)
    lead_time_change_days = st.slider("Lead time change (days)", -10, 30, 0, 1)
    otif_change_pct       = st.slider("Supplier OTIF change (%)",-30, 10, 0, 5)
    contract_active       = st.checkbox("Contract constraints active", value=False)
    moq_waived            = st.checkbox("MOQ waived", value=False)

    st.header("Thresholds")
    low_coverage_days = st.number_input("Low coverage (days)", min_value=1.0, value=14.0)
    poor_otif         = st.number_input("Poor OTIF threshold", 0.0, 1.0, 0.85, 0.01)


# =============================================================================
# HEADER
# =============================================================================

col1, col2, col3 = st.columns([1, 2, 3])
with col1:
    st.image("PlanSignal_light_.PNG", width=220)
st.markdown("<p style='margin-top:-15px;font-style:italic;'>Where planning data becomes decisions.</p>", unsafe_allow_html=True)
st.caption("Use the sidebar ← to adjust risk weights, run scenarios and set thresholds.")


# =============================================================================
# FILE UPLOADERS
# =============================================================================

with st.expander("📂 Upload data files", expanded=True):
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded           = st.file_uploader("SKU planning data (required)",          type=["csv","xlsx"], key="main_upload_v2")
        history_uploaded   = st.file_uploader("Forecast vs actual history (optional)",  type=["csv","xlsx"], key="history_upload")
        margin_uploaded    = st.file_uploader("Margin data (optional)",                 type=["csv","xlsx"], key="margin_upload")
    with col_up2:
        supplier_uploaded  = st.file_uploader("Supplier data (optional)",               type=["csv","xlsx"], key="supplier_upload")
        inventory_uploaded = st.file_uploader("Inventory data (optional)",              type=["csv","xlsx"], key="inventory_upload")
        leadtime_uploaded  = st.file_uploader("Lead time data (optional)",              type=["csv","xlsx"], key="leadtime_upload")
        contract_uploaded  = st.file_uploader("Contract data (optional)")      

template_df  = build_template()
template_csv = template_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV template",
    template_csv,
    "planning_risk_template.csv",
    "text/csv",
)

st.markdown("---")


# =============================================================================
# LOAD MAIN SKU FILE
# =============================================================================

if uploaded is None:
    st.info("No file uploaded — showing sample data.")
    df = template_df.copy()
else:
    try:
        temp_df = _read_file(uploaded)
        temp_df = _std_cols(temp_df)
        temp_df.rename(columns=RENAME_MAIN, inplace=True)

        missing = [c for c in ["sku","forecast","actual","volume"] if c not in temp_df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}. Showing sample data.")
            df = template_df.copy()
        else:
            df = temp_df.copy()
            df["sku"] = normalize_sku(df["sku"])
            st.success(f"SKU data loaded — {len(df):,} rows")
    except Exception as e:
        st.error(f"Could not read SKU file: {e}")
        df = template_df.copy()


# =============================================================================
# OPTIONAL DATA LOADERS
# =============================================================================

if margin_uploaded:
    margin_df = _load_optional(
        margin_uploaded,
        {
            "sku":"sku","product_code":"sku","material":"sku","item":"sku",
            "material_number":"sku",    # ← add this
            "material_no":"sku",        # ← and this (common SAP variant)
            "gross_margin":"margin","grossmargin":"margin",
            "margin_%":"margin","margin_pct":"margin","margin":"margin",
        },
        ["sku","margin"], "Margin data",
    )
    if margin_df is not None:
        margin_df["margin"] = pd.to_numeric(clean_number(margin_df["margin"]), errors="coerce")
        dups = margin_df[margin_df["sku"].duplicated(keep=False)]["sku"].unique()
        if len(dups):
            st.error(f"Duplicate SKUs in margin file: {', '.join(dups[:5])}")
        else:
            df = df.merge(margin_df[["sku","margin"]], on="sku", how="left", suffixes=("","_m"))
            if "margin_m" in df.columns:
                df["margin"] = df["margin_m"].combine_first(df["margin"])
                df.drop(columns=["margin_m"], inplace=True)

supplier_df = None
if supplier_uploaded:
    supplier_df = _load_optional(
        supplier_uploaded,
        {
            "sku":"sku","product_code":"sku","material":"sku","item":"sku",
            "supplier":"supplier","vendor":"supplier",
            "vendor_name":"supplier","supplier_name":"supplier",
            "supplier_otif":"supplier_otif","otif":"supplier_otif",
            "vendor_otif":"supplier_otif","otif_%":"supplier_otif",
        },
        ["sku","supplier_otif"], "Supplier data",
    )
    if supplier_df is not None:
        supplier_df["supplier_otif"] = pd.to_numeric(
            clean_number(supplier_df["supplier_otif"]), errors="coerce"
        )
        supplier_df["supplier_otif"] = supplier_df["supplier_otif"].apply(
            lambda x: x / 100 if pd.notna(x) and x > 1.0 else x
        )
        merge_cols = ["sku","supplier_otif"]
        if "supplier" in supplier_df.columns:
            merge_cols.append("supplier")
        df = df.merge(supplier_df[merge_cols], on="sku", how="left", suffixes=("","_s"))
        if "supplier_otif_s" in df.columns:
            df["supplier_otif"] = df["supplier_otif_s"].combine_first(df["supplier_otif"])
            df.drop(columns=["supplier_otif_s"], inplace=True)

if inventory_uploaded:
    inv_df = _load_optional(
        inventory_uploaded,
        {
            "sku":"sku","product_code":"sku","material":"sku","item":"sku",
            "inventory_on_hand":"inventory_on_hand","stock_on_hand":"inventory_on_hand",
            "inventory":"inventory_on_hand","on_hand":"inventory_on_hand",
            "safety_stock":"safety_stock","safetystock":"safety_stock","ss":"safety_stock",
        },
        ["sku","inventory_on_hand","safety_stock"], "Inventory data",
    )
    if inv_df is not None:
        for c in ["inventory_on_hand","safety_stock"]:
            inv_df[c] = pd.to_numeric(clean_number(inv_df[c]), errors="coerce")
        dups = inv_df[inv_df["sku"].duplicated(keep=False)]["sku"].unique()
        if len(dups):
            st.error(f"Duplicate SKUs in inventory file: {', '.join(dups[:5])}")
        else:
            df = df.merge(
                inv_df[["sku","inventory_on_hand","safety_stock"]],
                on="sku", how="left", suffixes=("","_i"),
            )
            for c in ["inventory_on_hand","safety_stock"]:
                if f"{c}_i" in df.columns:
                    df[c] = df[f"{c}_i"].combine_first(df[c])
                    df.drop(columns=[f"{c}_i"], inplace=True)

lt_df = None

if leadtime_uploaded:
    lt_df = _load_optional(
        leadtime_uploaded,
        {
            "sku":"sku","product_code":"sku","material":"sku","item":"sku",
            "lead_time":"lead_time_days","leadtime":"lead_time_days",
            "lead_time_days":"lead_time_days","leadtime_days":"lead_time_days",
            "replenishment_lead_time":"lead_time_days","lt":"lead_time_days",
        },
        ["sku","lead_time_days"], "Lead time data",
    )
    if lt_df is not None:
        lt_df["lead_time_days"] = pd.to_numeric(
            clean_number(lt_df["lead_time_days"]), errors="coerce"
        )
        bad = lt_df["lead_time_days"].isna().sum()
        if bad:
            st.warning(f"{bad} lead time row(s) could not be converted.")
        has_date = "date" in lt_df.columns
        if not has_date:
            dups = lt_df[lt_df["sku"].duplicated(keep=False)]["sku"].unique()
            if len(dups):
                st.error(
                    f"Duplicate SKUs in lead time file: {', '.join(dups[:5])}"
                )
        else:
            if "date" in lt_df.columns:
                lt_df["date"] = pd.to_datetime(lt_df["date"], errors="coerce")
                lt_latest = (
                    lt_df.sort_values("date")
                    .groupby("sku")["lead_time_days"]
                    .last()
                    .reset_index()
                )
                df = df.merge(lt_latest, on="sku", how="left", suffixes=("","_l"))
            if "lead_time_days_l" in df.columns:
                df["lead_time_days"] = df["lead_time_days_l"].combine_first(df["lead_time_days"])
                df.drop(columns=["lead_time_days_l"], inplace=True)
            else:
                df = df.merge(lt_df[["sku","lead_time_days"]], on="sku", how="left", suffixes=("","_l"))
                if "lead_time_days_l" in df.columns:
                    df["lead_time_days"] = df["lead_time_days_l"].combine_first(df["lead_time_days"])
                    df.drop(columns=["lead_time_days_l"], inplace=True)

if contract_uploaded:
    contract_df = _load_optional(
        contract_uploaded,
        {"sku":"sku","moq":"moq","contract_volume":"contract_volume","penalty_eur":"penalty_eur"},
        ["sku","moq"], "Contract data",
    )
    if contract_df is not None:
        for c in ["moq","contract_volume","penalty_eur"]:
            if c in contract_df.columns:
                contract_df[c] = pd.to_numeric(contract_df[c], errors="coerce").fillna(0)
        df = df.merge(contract_df, on="sku", how="left")

history_df = None
if history_uploaded is not None:
    try:
        history_df = _read_file(history_uploaded)
        history_df = _std_cols(history_df)
        history_df.rename(columns={
            "product_code":"sku","product code":"sku",
            "date":"date","month":"date",
        }, inplace=True)
        if not {"sku","actual","forecast"}.issubset(history_df.columns):
            st.error("History file must contain: sku, actual, forecast.")
            history_df = None
        else:
            history_df["sku"]      = normalize_sku(history_df["sku"])
            history_df["actual"]   = pd.to_numeric(clean_number(history_df["actual"]),   errors="coerce")
            history_df["forecast"] = pd.to_numeric(clean_number(history_df["forecast"]), errors="coerce")
            st.success(f"Forecast history loaded — {len(history_df):,} rows")
    except Exception as e:
        st.error(f"History file error: {e}")
        history_df = None


# =============================================================================
# DEFAULT FILLS
# =============================================================================

for col, default in {
    "margin":            1.0,
    "inventory_on_hand": 0.0,
    "safety_stock":      0.0,
    "lead_time_days":   14.0,
    "supplier_otif":     1.0,
}.items():
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)


# =============================================================================
# COMPUTE METRICS & ASSIGN ACTIONS
# =============================================================================

thresholds = RiskThresholds(low_coverage_days=low_coverage_days, poor_otif=poor_otif)

metrics_df = compute_metrics(
    df,
    demand_weight, supply_weight, inventory_weight,
    history_df=history_df,
    demand_change_pct=demand_change_pct,
    lead_time_change_days=lead_time_change_days,
    otif_change_pct=otif_change_pct, lt_df=lt_df
)

result_df = assign_action(metrics_df, thresholds, contract_active, moq_waived)

result_df["action_status"] = result_df["sku"].map(
    lambda s: st.session_state["action_status"].get(s, "Open")
)

# Portfolio-level WMAPE
portfolio_wmape = compute_wmape_portfolio(history_df)


# =============================================================================
# PORTFOLIO HEALTH SUMMARY
# =============================================================================

total_skus        = len(result_df)
high_cutoff       = result_df["business_risk"].quantile(0.75)
mid_cutoff        = result_df["business_risk"].quantile(0.40)
high_risk_count   = int((result_df["business_risk"] >= high_cutoff).sum())
total_risk        = float(result_df["business_risk"].sum())
risk_top3         = float(result_df.nlargest(3,"business_risk")["business_risk"].sum())
risk_concentration = risk_top3 / total_risk if total_risk > 0 else 0
total_excess_val  = float(result_df["excess_stock_value"].sum())
open_actions      = int((result_df["action_status"] == "Open").sum())
expedite_count    = int((result_df["recommended_action"] == "EXPEDITE").sum())

with st.container(border=True):
    st.markdown("### Portfolio Health")
    st.caption(
        "At-a-glance view of risk, concentration, working capital exposure, and open actions. "
        "Forecast accuracy shown as WMAPE (volume-weighted)."
    )
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Total SKUs",        total_skus)
    c2.metric("High-Risk SKUs",    high_risk_count,
              delta=f"{high_risk_count/total_skus:.0%} of portfolio", delta_color="inverse")
    c3.metric("Expedite Now",      expedite_count,
              delta="coverage <= 3d", delta_color="inverse")
    c4.metric("Risk Exposure",     f"EUR {total_risk:,.0f}")
    c5.metric("Top 3 Share",       f"{risk_concentration:.1%}",
              delta="concentration", delta_color="inverse")
    c6.metric("Excess Stock",      f"EUR {total_excess_val:,.0f}",
              delta="working capital", delta_color="inverse")
    c7.metric("Open Actions",      open_actions,
              delta="need attention", delta_color="inverse")
    c8.metric(
        "Portfolio WMAPE",
        f"{portfolio_wmape:.1f}%" if portfolio_wmape is not None else "N/A",
        delta="forecast accuracy" if portfolio_wmape is not None else "upload history",
        delta_color="inverse" if portfolio_wmape is not None else "off",
    )

st.markdown("---")


# =============================================================================
# PRIORITY SKU TABLE
# =============================================================================

max_top_n     = max(1, min(200, len(result_df)))
default_top_n = min(50, max_top_n)
top_n = (
    st.slider("SKUs to display", min(5, max_top_n), max_top_n, default_top_n)
    if max_top_n > 5 else max_top_n
)

st.subheader("Priority SKUs")
st.markdown(
    '<div class="section-note">'
    'Ranked by business risk. WMAPE shown per SKU — higher = less accurate forecast. '
    'Update Status inline.'
    '</div>',
    unsafe_allow_html=True,
)

display_source = {
    "sku":               "SKU",
    "recommended_action":"Action",
    "action_status":     "Status",
    "risk_drivers":      "Risk Drivers",
    "business_risk":     "Business Risk Index",
    "revenue_at_risk":   "Revenue at Risk (€)",
    "demand_trend":      "Trend",
    "wmape":             "WMAPE (%)",
    "excess_stock_value":"Excess Stock (EUR)",
    "coverage_days":     "Coverage (days)",
    "lead_time_days":    "Lead Time (days)",
    "supplier_otif":     "OTIF (%)",
    "bias":              "Bias",
    "abs_error":         "Abs Error",
}

table_df = result_df.head(top_n)[[c for c in display_source if c in result_df.columns]].copy()
table_df.rename(columns=display_source, inplace=True)
table_df["Action"]             = table_df["Action"].map(format_action)
table_df["Business Risk Index"] = table_df["Business Risk Index"].round(0)
table_df["Excess Stock (EUR)"] = table_df["Excess Stock (EUR)"].round(0)
table_df["Coverage (days)"]    = table_df["Coverage (days)"].round(1)
table_df["Lead Time (days)"]   = table_df["Lead Time (days)"].round(0)
table_df["OTIF (%)"]           = (table_df["OTIF (%)"] * 100).round(1)
table_df["Bias"]               = table_df["Bias"].round(0)
table_df["Abs Error"]          = table_df["Abs Error"].round(0)
if "WMAPE (%)" in table_df.columns:
    table_df["WMAPE (%)"] = table_df["WMAPE (%)"].round(1)
if "Days to Stockout" in table_df.columns:
    table_df["Days to Stockout"] = table_df["Days to Stockout"].round(1)
if "Revenue at Risk (€)" in table_df.columns:
    table_df["Revenue at Risk (€)"] = table_df["Revenue at Risk (€)"].round(0)

edited = st.data_editor(
    table_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Status": st.column_config.SelectboxColumn(
            "Status",
            options=ACTION_STATUS_OPTIONS,
            required=True,
            help="Update to track progress on each action.",
        ),
        "Risk (EUR)": st.column_config.NumberColumn(format="%,.0f"),
        "Excess Stock (EUR)": st.column_config.NumberColumn(format="%,.0f"),
        "Days to Stockout": st.column_config.NumberColumn(
            help="Days until inventory reaches zero at current demand rate.",
            format="%.1f",
        ),
        "Revenue at Risk (€)": st.column_config.NumberColumn(
            help="Financial exposure if stockout occurs. Margin × units short × daily demand.",
            format="%,.0f",
        ),
        "WMAPE (%)": st.column_config.NumberColumn(
            help="Volume-weighted forecast error. Lower = more accurate.",
            format="%.1f%%",
        ),
        "Trend": st.column_config.TextColumn(
            help="3-period demand slope: up = growing, stable, down = declining"
        ),
    },
    disabled=[c for c in table_df.columns if c != "Status"],
)

for _, row in edited.iterrows():
    st.session_state["action_status"][row["SKU"]] = row["Status"]


# =============================================================================
# EXCEL EXPORT
# =============================================================================

export_result = result_df.copy()
export_result["action_status"] = export_result["sku"].map(
    lambda s: st.session_state["action_status"].get(s, "Open")
)
excel_bytes = build_excel_export(export_result)

st.download_button(
    "Export Priority Actions (Excel)",
    data=excel_bytes,
    file_name=f"priority_actions_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown("---")


# =============================================================================
# RISK EXPOSURE CHART
# =============================================================================

with st.expander("Risk Exposure", expanded=False):
    st.markdown(
        '<div class="section-note">Top SKUs by business risk. Screenshot-ready for S&OP reviews.</div>',
        unsafe_allow_html=True,
    )
    chart_df = (
        result_df[["sku","business_risk"]]
        .head(top_n)
        .sort_values("business_risk", ascending=False)
        .rename(columns={"sku":"SKU","business_risk":"Risk (EUR)"})
    )
    st.bar_chart(chart_df, x="SKU", y="Risk (EUR)", horizontal=True)

st.markdown("---")


# =============================================================================
# SUPPLIER RISK CLUSTERING
# =============================================================================

if "supplier" in result_df.columns:
    st.subheader("Supplier Risk Clustering")
    st.markdown(
        '<div class="section-note">'
        'High-risk SKUs grouped by supplier. Enables supplier-level conversations '
        'instead of item-by-item firefighting.'
        '</div>',
        unsafe_allow_html=True,
    )

    high_risk_skus = result_df[result_df["business_risk"] >= high_cutoff].copy()

    if high_risk_skus.empty or high_risk_skus["supplier"].isna().all():
        st.info("No supplier information available for high-risk SKUs.")
    else:
        cluster = (
            high_risk_skus
            .groupby("supplier", dropna=True)
            .agg(
                high_risk_skus=("sku",           "count"),
                total_risk    =("business_risk", "sum"),
                avg_otif      =("supplier_otif", "mean"),
                avg_coverage  =("coverage_days", "mean"),
            )
            .sort_values("total_risk", ascending=False)
            .reset_index()
        )
        cluster["avg_otif"]    = (cluster["avg_otif"] * 100).round(1)
        cluster["avg_coverage"] = cluster["avg_coverage"].round(1)
        cluster["total_risk"]  = cluster["total_risk"].round(0)
        cluster.columns = [
            "Supplier","High-Risk SKUs","Total Risk (EUR)","Avg OTIF (%)","Avg Coverage (days)"
        ]
        st.dataframe(cluster, use_container_width=True, hide_index=True)

    st.markdown("---")


# =============================================================================
# SKU DRILL-DOWN
# =============================================================================

st.subheader("SKU Drill-Down")

selected_sku = st.selectbox("Select SKU", result_df["sku"].tolist())
selected     = result_df[result_df["sku"] == selected_sku].iloc[0]

# WMAPE for this SKU from history
sku_wmape   = None
sku_history = None

if history_df is not None:
    sku_history = history_df[history_df["sku"] == selected_sku].copy()
    if not sku_history.empty:
        sku_wmape = compute_wmape_sku(sku_history["actual"], sku_history["forecast"])

# KPI rows
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Business Risk Index", f"EUR {selected['business_risk']:,.0f}",
          help="Composite risk score weighted by unit margin and volume. Ranks priority — not a precise financial forecast.")
k2.metric("Action",            selected["recommended_action"])
k3.metric("Status",            st.session_state["action_status"].get(selected_sku, "Open"))
k4.metric("Coverage (days)",   f"{selected['coverage_days']:.1f}")
k5.metric("Supplier OTIF",     f"{selected['supplier_otif'] * 100:.1f}%")
k6.metric("WMAPE",             f"{sku_wmape:.1f}%" if sku_wmape is not None else "N/A",
          help="Volume-weighted forecast error for this SKU across all history periods.")

k7, k8, k9 = st.columns(3)
k7.metric("Revenue at Risk",    f"EUR {selected['revenue_at_risk']:,.0f}",
          help="Estimated margin exposure from shortage risk before replenishment arrives. Based on projected shortfall units × unit margin.")
k8.metric("Excess Stock (EUR)", f"EUR {selected['excess_stock_value']:,.0f}")
k9.metric("Demand Trend",      selected.get("demand_trend", "→"))

# Risk drivers
reasons       = explain_sku_risk(selected)
action        = selected.get("recommended_action","").upper()
clean_reasons = " • ".join(reasons)

st.markdown("#### Risk Drivers")
if selected["coverage_days"] <= 3:
    st.error(f"EXPEDITE REQUIRED — {clean_reasons}")
elif action == "OPTIMISE INVENTORY":
    st.warning(f"{clean_reasons}")
elif action == "REVIEW CONTRACT":
    st.warning(f"Contract constraint active — {clean_reasons}")
elif selected["inventory_on_hand"] < selected["safety_stock"] or selected["supplier_otif"] < 0.80:
    st.error(f"{clean_reasons}")
elif selected["coverage_days"] <= 14 or selected["supplier_otif"] < 0.90 or selected["lead_time_days"] > 30:
    st.warning(f"{clean_reasons}")
elif action == "REVIEW FORECAST":
    st.warning(f"{clean_reasons}")
else:
    st.success(f"{clean_reasons}")

# Metric breakdown
explain_data = {
    "Metric": [
        "Bias","Absolute Error","WMAPE (%)","Volatility CV",
        "Lead Time (days)","Inventory on Hand","Safety Stock",
        "Excess Stock (EUR)","Margin","Demand Trend", "LT Trend"
    ],
    "Value": [
        round(float(selected["bias"]),              2),
        round(float(selected["abs_error"]),          2),
        round(float(sku_wmape), 1) if sku_wmape is not None else "N/A",
        round(float(selected["volatility_cv"]),      3),
        round(float(selected["lead_time_days"]),     1),
        round(float(selected["inventory_on_hand"]),  1),
        round(float(selected["safety_stock"]),       1),
        round(float(selected["excess_stock_value"]), 0),
        round(float(selected["margin"]),             2),
        selected.get("demand_trend","→"),
        selected.get("leadtime_trend", "🟢 Stable")
    ],
}
with st.expander("📊 Metric breakdown", expanded=False):
    st.dataframe(pd.DataFrame(explain_data), use_container_width=True, hide_index=True)

# Forecast vs Actual chart
if sku_history is not None and not sku_history.empty and "date" in sku_history.columns:
    sku_history["date"]  = pd.to_datetime(sku_history["date"], errors="coerce")
    sku_history["error"] = sku_history["actual"] - sku_history["forecast"]
    sku_history = (
        sku_history.dropna(subset=["date","forecast","actual"])
        .sort_values("date")
    )
    if not sku_history.empty:
        st.markdown(f"#### Forecast vs Actual — {selected_sku}")
        st.line_chart(sku_history.set_index("date")[["forecast","actual","error"]])
else:
    st.info(
        "Upload a forecast history file with a date column to see the trend chart. "
        "Required columns: sku, date, forecast, actual."
    )


# =============================================================================
# MODEL EXPLAINER
# =============================================================================

with st.expander("How the model works"):
    st.markdown("""
**WMAPE (Weighted Mean Absolute Percentage Error)**
`sum(|actual - forecast|) / sum(actual) x 100`
High-volume SKU errors dominate the score. A small SKU missing by 50% barely registers.
A large SKU missing by 10% shows up strongly. This is the pharma and chemical industry standard.

**Demand risk** — forecast error x volume, scaled by demand volatility and WMAPE.

**Supply risk** — (1 - OTIF) x lead time. Poor reliability + long lead time = high exposure.

**Inventory risk** — shortage below safety stock weighted heavily.
Excess above 2x safety stock weighted at 25% (working capital concern).

**Business impact** — composite risk score x unit margin x volume.

**Action logic:**

| Action | Trigger |
|---|---|
| EXPEDITE | Coverage <= 3 days |
| SUPPLY ISSUE | Replenishment at risk — stock, supplier, or lead time cannot guarantee continuity |
| REVIEW CONTRACT | Contract active + outstanding call-off + excess cover |
| OPTIMISE INVENTORY | Coverage > 60 days + stock > 2x safety stock |
| MONITOR | No urgent trigger detected |

**Demand trend** — 3-period linear slope from hist_1/hist_2/hist_3.
Up = >5% positive slope relative to mean. Down = >5% negative. Stable = within range.

**Excess stock (EUR)** — units above 2x safety stock x unit margin.
    """)
