import os
import json
from datetime import datetime

import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import gdown

# ===== CONFIG =====
DB_PATH = "analytics.db"
FILE_ID = "1r364Oitl8CnQ7-13e2egGOQ8mLwcv-JD"
URL = f"https://drive.google.com/uc?id={FILE_ID}"
MIN_DB_BYTES = 1000  # sanity check

# ===== CACHE COMPAT LAYER =====
# Use cache_resource when available; otherwise fall back to experimental_singleton (older Streamlit).
def cache_resource_decorator():
    if hasattr(st, "cache_resource"):
        return st.cache_resource
    if hasattr(st, "experimental_singleton"):
        return st.experimental_singleton
    # very old Streamlit fallback
    return st.cache  # type: ignore

cache_resource = cache_resource_decorator()

# ===== REFRESH BUTTON =====
if st.sidebar.button("🔄 Refresh DB"):
    # Try to remove local DB (okay if it fails, we just warn)
    try:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
    except Exception as e:
        st.sidebar.warning(f"Couldn't delete local DB: {e}")

    # Clear whichever cache type exists
    if hasattr(st, "cache_resource"):
        st.cache_resource.clear()
    elif hasattr(st, "experimental_singleton"):
        st.experimental_singleton.clear()
    else:
        st.caching.clear_cache()

    # Rerun (prefer st.rerun if available)
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# ===== DOWNLOADER =====
def _download_db_if_needed():
    need_download = (not os.path.exists(DB_PATH)) or (os.path.getsize(DB_PATH) < MIN_DB_BYTES)
    if need_download:
        st.sidebar.info("Downloading latest database... ⏳")
        gdown.download(URL, DB_PATH, quiet=False, fuzzy=False)

    # Verify after download
    if (not os.path.exists(DB_PATH)) or (os.path.getsize(DB_PATH) < MIN_DB_BYTES):
        st.error("❌ Database download failed! Check the Google Drive link or permissions.")
        st.stop()

# ===== CACHED CONNECTION =====
@cache_resource(show_spinner=False)
def get_connection():
    _download_db_if_needed()
    # IMPORTANT: Do NOT close this connection later; it is cached.
    return duckdb.connect(DB_PATH, read_only=False)

con = get_connection()

# Show last updated timestamp in sidebar (if file exists)
if os.path.exists(DB_PATH):
    ts = datetime.fromtimestamp(os.path.getmtime(DB_PATH))
    st.sidebar.caption(f"📦 DB last updated: {ts.strftime('%Y-%m-%d %H:%M:%S')}")

# ===== HELPERS =====
def calculate_stats(df, cols):
    stats_data = []
    for col in cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        stats_data.append({
            "Metric": col,
            "Mean": round(series.mean(), 4),
            "Median": round(series.median(), 4),
            "Min": round(series.min(), 4),
            "Max": round(series.max(), 4),
            "Std Dev": round(series.std(), 4),
            "% Positive": round((series > 0).mean() * 100, 2),
            "% Negative": round((series < 0).mean() * 100, 2),
        })
    return pd.DataFrame(stats_data)

def plot_with_zero_coloring(df, x_col, y_cols, chart_title):
    fig = go.Figure()

    def get_segments(x, y):
        segments = []
        current_x, current_y = [], []
        idx = y.first_valid_index()
        if idx is None:
            return segments
        prev_sign = np.sign(y.loc[idx])
        for xi, yi in zip(x, y):
            if pd.isna(yi):
                if current_x:
                    segments.append((current_x, current_y, prev_sign))
                    current_x, current_y = [], []
                prev_sign = None
                continue
            sign = np.sign(yi)
            if prev_sign is None or sign == prev_sign:
                current_x.append(xi)
                current_y.append(yi)
                prev_sign = sign
            else:
                segments.append((current_x, current_y, prev_sign))
                current_x, current_y = [xi], [yi]
                prev_sign = sign
        if current_x:
            segments.append((current_x, current_y, prev_sign))
        return segments

    colors_map = {1: "green", 0: "gray", -1: "red"}

    for col in y_cols:
        x = df[x_col]
        y = df[col]
        segments = get_segments(x, y)
        for (seg_x, seg_y, sign) in segments:
            color = colors_map.get(sign, "black")
            fig.add_trace(go.Scatter(
                x=seg_x,
                y=seg_y,
                mode='lines+markers',
                name=f"{col} ({'pos' if sign > 0 else 'neg' if sign < 0 else 'zero'})",
                line=dict(color=color, width=2),
                marker=dict(size=4),
                showlegend=True
            ))

    fig.update_layout(
        title=chart_title,
        xaxis_title="Date",
        yaxis_title="Value",
        template='plotly_white',
        height=400,
        legend_title="Metrics"
    )
    st.plotly_chart(fig, use_container_width=True)

# ===== DATA LOAD =====
query = "SELECT * FROM rpt_aus200"
df = con.execute(query).fetchdf()
df["date"] = pd.to_datetime(df["date"])

# ===== FILTERS =====
st.sidebar.header("Filters")
filter_cols = st.sidebar.multiselect("Select columns to filter", df.columns)

for col in filter_cols:
    if df[col].dtype == "object":
        options = df[col].dropna().unique().tolist()
        selected = st.sidebar.multiselect(f"Filter {col}", options)
        if selected:
            df = df[df[col].isin(selected)]

    elif pd.api.types.is_numeric_dtype(df[col]):
        min_val, max_val = float(df[col].min()), float(df[col].max())
        selected = st.sidebar.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
        df = df[(df[col] >= selected[0]) & (df[col] <= selected[1])]

    elif pd.api.types.is_datetime64_any_dtype(df[col]):
        min_date, max_date = df[col].min(), df[col].max()
        date_range = st.sidebar.date_input(f"Filter {col}", [min_date, max_date])
        if len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df[(df[col] >= start) & (df[col] <= end)]

# ===== UI =====
st.title("AUS200 historical moves")
st.dataframe(df)

# Save matched dates (last 20)
matched_dates = df["date"].dt.date.unique().astype(str).tolist()[-20:]
output_file = "matched_dates.json"
with open(output_file, "w") as f:
    json.dump(matched_dates, f)
st.sidebar.success(f"Matched dates saved to {output_file}")

numeric_cols = df.select_dtypes(include="number").columns.tolist()

# === Chart 1 ===
st.header("Chart 1: Select metrics")
cols_chart1 = st.multiselect("Columns for chart 1", options=numeric_cols, default=[], key="c1")
if cols_chart1:
    st.subheader("Statistics for Chart 1 Metrics")
    st.table(calculate_stats(df, cols_chart1))
    plot_with_zero_coloring(df, "date", cols_chart1, "Chart 1: Selected Metrics with Zero-Based Coloring")

# === Chart 2 ===
st.header("Chart 2: Select metrics")
cols_chart2 = st.multiselect("Columns for chart 2", options=numeric_cols, default=[], key="c2")
if cols_chart2:
    st.subheader("Statistics for Chart 2 Metrics")
    st.table(calculate_stats(df, cols_chart2))
    plot_with_zero_coloring(df, "date", cols_chart2, "Chart 2: Selected Metrics with Zero-Based Coloring")

# ===== CANDLE CHARTS =====
st.sidebar.header("Candlestick Charts")
if st.sidebar.button("Print Charts") and matched_dates:
    TF = "5m"
    TABLE_NAME = f"main.stg_aus_{TF}"

    # Use a short-lived separate connection
    con2 = duckdb.connect(DB_PATH)

    date_list_str = ",".join([f"'{d}'" for d in matched_dates])
    q2 = f"""
        SELECT time, open, high, low, close
        FROM {TABLE_NAME}
        WHERE CAST(time AS DATE) IN ({date_list_str})
          AND EXTRACT(HOUR FROM time) BETWEEN 9 AND 16
        ORDER BY time
    """
    df_candles = con2.execute(q2).df()
    con2.close()

    for date_str in matched_dates:
        daily_data = df_candles[df_candles["time"].dt.date.astype(str) == date_str]

        fig = go.Figure(data=[go.Candlestick(
            x=daily_data["time"],
            open=daily_data["open"],
            high=daily_data["high"],
            low=daily_data["low"],
            close=daily_data["close"],
            increasing_line_color="green",
            decreasing_line_color="red",
        )])

        fig.update_layout(
            title=f"AUS200 {TF} Intraday: {date_str} (9AM–4PM NSW)",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
        )

        st.plotly_chart(fig, use_container_width=True)

# NOTE: Do NOT close `con` here — it is cached and reused across reruns.
