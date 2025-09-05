import duckdb
import pandas as pd
import streamlit as st
import json
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import plotly.io as pio
import gdown

# Use Streamlit's page configuration for a wider layout
st.set_page_config(layout='wide')

# ===== DATABASE SETUP =====
DB_PATH = "analytics.db"
FILE_ID = "1r364Oitl8CnQ7-13e2egGOQ8mLwcv-JD"

# Download the DB if it doesn't exist
if not os.path.exists(DB_PATH):
    st.sidebar.info("Downloading database... please wait ⏳")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, DB_PATH, quiet=False)
    
# Connect to the downloaded DB
con = duckdb.connect(DB_PATH)

# ===== FUNCTION DEFINITIONS =====
@st.cache_data
def load_data(query):
    """Caches the result of a DuckDB query to prevent re-running."""
    return con.execute(query).fetchdf()

@st.cache_data
def calculate_stats(df, cols):
    """Calculates summary statistics for a given DataFrame and columns."""
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
    """Creates a line chart with segments colored based on positive/negative values."""
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

def plot_high_low_occurrences(df):
    """
    Generates a bar chart showing the occurrences of rth_high and rth_low times.
    """
    if 'rth_high' not in df.columns or 'rth_low' not in df.columns:
        st.error("The dataframe must contain 'rth_high' and 'rth_low' columns.")
        return

    # Combine rth_high and rth_low into a single Series and count occurrences
    high_low_counts = pd.concat([df['rth_high'], df['rth_low']]).value_counts().reset_index()
    high_low_counts.columns = ['time', 'occurrences']
    
    # Sort the dataframe by time to ensure consistent ordering on the x-axis
    high_low_counts['time'] = pd.to_datetime(high_low_counts['time'], format='%H:%M:%S').dt.time
    high_low_counts = high_low_counts.sort_values(by='time')

    # Create the bar chart using Plotly Express
    fig = px.bar(
        high_low_counts,
        x='time',
        y='occurrences',
        title="Occurrences of RTH High and Low Times"
    )

    # Customize the chart layout
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Number of Occurrences",
        title_x=0.5,
        xaxis={'type': 'category'} # Treat x-axis as categories to avoid gaps
    )
    
    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# ===== RUN QUERY =====
query = "SELECT * FROM rpt_aus200"
df = load_data(query)
df['date'] = pd.to_datetime(df['date'])

# ===== STREAMLIT FILTERS =====
st.sidebar.header("Filters")

# Let user pick which columns to filter
filter_cols = st.sidebar.multiselect("Select columns to filter", df.columns)

# Build filters dynamically
for col in filter_cols:
    if df[col].dtype == 'object':
        options = df[col].dropna().unique().tolist()
        selected = st.sidebar.multiselect(f"Filter {col}", options)
        if selected:
            df = df[df[col].isin(selected)]

    elif pd.api.types.is_numeric_dtype(df[col]):
        min_val, max_val = float(df[col].min()), float(df[col].max())
        selected = st.sidebar.slider(
            f"Filter {col}",
            min_val,
            max_val,
            (min_val, max_val)
        )
        df = df[(df[col] >= selected[0]) & (df[col] <= selected[1])]

    elif pd.api.types.is_datetime64_any_dtype(df[col]):
        min_date, max_date = df[col].min(), df[col].max()
        date_range = st.sidebar.date_input(f"Filter {col}", [min_date, max_date])
        if len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df[(df[col] >= start) & (df[col] <= end)]

# ===== STREAMLIT UI =====
st.title("AUS200 historical moves")
st.dataframe(df)

# Call the new function to plot the RTH high/low occurrences
st.header("RTH High and Low Time Occurrences")
plot_high_low_occurrences(df)

# Get all unique dates, sort them in descending order, and then get the top 20
unique_dates = df['date'].dt.date.unique()
sorted_dates = pd.to_datetime(unique_dates).sort_values(ascending=False)
matched_dates = sorted_dates[:20].strftime('%Y-%m-%d').tolist()

# Use Streamlit's session state instead of writing to a file
if 'matched_dates' not in st.session_state:
    st.session_state['matched_dates'] = []

st.session_state['matched_dates'] = matched_dates
st.sidebar.success(f"Found {len(matched_dates)} dates for charts.")


numeric_cols = df.select_dtypes(include='number').columns.tolist()

# === Chart 1 ===
st.header("Chart 1: Select metrics")
cols_chart1 = st.multiselect("Columns for chart 1", options=numeric_cols, default=[], key="c1")
if cols_chart1:
    st.subheader("Statistics for Chart 1 Metrics")
    st.table(calculate_stats(df, cols_chart1))
    plot_with_zero_coloring(df, 'date', cols_chart1, "Chart 1: Selected Metrics with Zero-Based Coloring")

# === Chart 2 ===
st.header("Chart 2: Select metrics")
cols_chart2 = st.multiselect("Columns for chart 2", options=numeric_cols, default=[], key="c2")
if cols_chart2:
    st.subheader("Statistics for Chart 2 Metrics")
    st.table(calculate_stats(df, cols_chart2))
    plot_with_zero_coloring(df, 'date', cols_chart2, "Chart 2: Selected Metrics with Zero-Based Coloring")

# ===== PRINT CHARTS IN BROWSER =====

st.sidebar.header("Candlestick Charts")
if st.sidebar.button("Print Charts") and st.session_state.get('matched_dates'):
    with st.spinner("Generating candlestick charts..."):
        TF = "5m"
        TABLE_NAME = f"main.stg_aus_{TF}"
        
        # Use a new connection for the specific query
        con2 = duckdb.connect(DB_PATH)

        date_list_str = ",".join([f"'{d}'" for d in st.session_state['matched_dates']])
        query = f"""
        SELECT
            time,
            open,
            high,
            low,
            close
        FROM {TABLE_NAME}
        WHERE
            CAST(time AS DATE) IN ({date_list_str})
            AND EXTRACT(HOUR FROM time) BETWEEN 9 AND 16
        ORDER BY time
        """
        df_candles = con2.execute(query).df()
        con2.close()

        for date_str in st.session_state['matched_dates']:
            daily_data = df_candles[df_candles['time'].dt.date.astype(str) == date_str]

            if not daily_data.empty:
                fig = go.Figure(data=[go.Candlestick(
                    x=daily_data['time'],
                    open=daily_data['open'],
                    high=daily_data['high'],
                    low=daily_data['low'],
                    close=daily_data['close'],
                    increasing_line_color='green',
                    decreasing_line_color='red'
                )])

                fig.update_layout(
                    title=f"AUS200 {TF} Intraday: {date_str} (9AM-4PM NSW)",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False
                )

                st.plotly_chart(fig, use_container_width=True)

con.close()