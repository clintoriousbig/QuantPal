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
st.set_page_config(layout='wide', page_title="AUS200 Dashboard", page_icon="📈")

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

def plot_high_low_separate_charts(df):
    """
    Generates two separate bar charts for RTH High and RTH Low occurrences.
    """
    if 'rth_high' not in df.columns or 'rth_low' not in df.columns:
        st.error("The dataframe must contain 'rth_high' and 'rth_low' columns.")
        return
    
    # Create two columns to display the charts side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # --- Plot 1: RTH High Occurrences (Green Bars) ---
        high_counts = df['rth_high'].value_counts().reset_index()
        high_counts.columns = ['time', 'occurrences']
        high_counts['time'] = pd.to_datetime(high_counts['time'], format='%H:%M:%S').dt.time
        high_counts = high_counts.sort_values(by='time')

        fig_high = px.bar(
            high_counts,
            x='time',
            y='occurrences',
            title="RTH High Time Occurrences",
        )
        fig_high.update_traces(marker_color='green')
        fig_high.update_layout(xaxis_title="Time", yaxis_title="Number of Occurrences", title_x=0.5, xaxis={'type': 'category'})
        st.plotly_chart(fig_high, use_container_width=True)
    
    with col2:
        # --- Plot 2: RTH Low Occurrences (Red Bars) ---
        low_counts = df['rth_low'].value_counts().reset_index()
        low_counts.columns = ['time', 'occurrences']
        low_counts['time'] = pd.to_datetime(low_counts['time'], format='%H:%M:%S').dt.time
        low_counts = low_counts.sort_values(by='time')

        fig_low = px.bar(
            low_counts,
            x='time',
            y='occurrences',
            title="RTH Low Time Occurrences",
        )
        fig_low.update_traces(marker_color='red')
        fig_low.update_layout(xaxis_title="Time", yaxis_title="Number of Occurrences", title_x=0.5, xaxis={'type': 'category'})
        st.plotly_chart(fig_low, use_container_width=True)

# ===== RUN QUERY =====
query = "SELECT * FROM rpt_aus200"
df = load_data(query)
df['date'] = pd.to_datetime(df['date'])

# Create a mapping from snake_case to human-readable names
filter_name_map = {
    'aus_gap_pct': 'AUS Gap (%)',
    'spx_gap_pct': 'SPX Gap (%)',
    'aus_prev_pct': 'AUS Prev Day (%)',
    'uk_prev_pct': 'UK Prev Day (%)',
    'bco_prev_pct': 'BCO Prev Day (%)',
    'xau_prev_pct': 'XAU Prev Day (%)',
    'spx_prev_pct': 'SPX Prev Day (%)',
    'aud_prev_pct': 'AUD Prev Day (%)',
    'rth_change_pts': 'RTH Change (pts)',
    'rth_open2high': 'RTH Open-to-High (%)',
    'rth_open2low': 'RTH Open-to-Low (%)',
    'rth_range': 'RTH Range',
    'change_pct': 'Daily Change (%)',
    'prev_up_wick': 'Prev Day Upper Wick (%)',
    'prev_low_wick': 'Prev Day Lower Wick (%)',
    'pct_9am': '9 AM Change (%)',
    'up_wick_9am': '9 AM Upper Wick (%)',
    'low_wick_9am': '9 AM Lower Wick (%)',
    'pct_10am': '10 AM Change (%)',
    'up_wick_10am': '10 AM Upper Wick (%)',
    'low_wick_10am': '10 AM Lower Wick (%)',
    'aus_950': '9:50 AM Move (%)',
    'aus_955': '9:55 AM Move (%)',
    'aus_1000': '10:00 AM Move (%)',
    'aus_1005': '10:05 AM Move (%)',
    'open2high_950': '9:50 AM Open-to-High (%)',
    'open2low_950': '9:50 AM Open-to-Low (%)',
    'pct_to_high_1000': '10:00 AM Open-to-High (%)',
    'pct_to_low_1000': '10:00 AM Open-to-Low (%)',
    'pct_change_1010': '10:10 AM Change (%)',
    'pct_to_high_1010': '10:10 AM Open-to-High (%)',
    'pct_to_low_1010': '10:10 AM Open-to-Low (%)',
    'pct_change_1020': '10:20 AM Change (%)',
    'pct_to_high_1020': '10:20 AM Open-to-High (%)',
    'pct_to_low_1020': '10:20 AM Open-to-Low (%)',
    'spx_950': 'SPX 9:50 AM Move (%)',
    'spx_10': 'SPX 10 AM Move (%)',
    'news': 'News Description',
    'news_impact': 'News Impact'
}

# Invert the map for lookups
name_filter_map = {v: k for k, v in filter_name_map.items()}

# Define filter groups
filter_groups = {
    "Open Gaps": ['aus_gap_pct', 'spx_gap_pct'],
    "Previous Day Metrics": ['aus_prev_pct', 'uk_prev_pct', 'bco_prev_pct', 'xau_prev_pct', 'spx_prev_pct', 'aud_prev_pct', 'prev_up_wick', 'prev_low_wick'],
    "RTH Metrics": ['rth_change_pts', 'rth_open2high', 'rth_open2low', 'rth_range'],
    "Daily Moves": ['change_pct'],
    "Early Morning Moves (5m/10m)": [
        'aus_950', 'aus_955', 'aus_1000', 'aus_1005',
        'open2high_950', 'open2low_950',
        'pct_to_high_1000', 'pct_to_low_1000',
        'pct_change_1010', 'pct_to_high_1010',
        'pct_to_low_1010',
        'pct_change_1020',
        'pct_to_high_1020',
        'pct_to_low_1020'
    ],
    "Hourly Moves": ['pct_9am', 'up_wick_9am', 'low_wick_9am', 'pct_10am', 'up_wick_10am', 'low_wick_10am'],
    "News": ['news', 'news_impact'],
    "SPX RTH Metrics": ['spx_950', 'spx_10']
}

# ===== STREAMLIT FILTERS (UPDATED) =====
st.sidebar.header("Filters")
# Create a copy of the dataframe to apply filters
filtered_df = df.copy()

# Loop through each filter group to create an expander
for group_name, group_cols in filter_groups.items():
    with st.sidebar.expander(group_name, expanded=False):
        for col in group_cols:
            if col in df.columns:
                human_name = filter_name_map[col]

                if df[col].dtype == 'object':
                    options = df[col].dropna().unique().tolist()
                    selected = st.multiselect(f"Filter {human_name}", options, key=f"multiselect_{col}")
                    if selected:
                        filtered_df = filtered_df[filtered_df[col].isin(selected)]
                
                elif pd.api.types.is_numeric_dtype(df[col]):
                    min_val, max_val = float(df[col].min()), float(df[col].max())
                    selected_range = st.slider(
                        f"Filter {human_name}",
                        min_val,
                        max_val,
                        (min_val, max_val),
                        key=f"slider_{col}"
                    )
                    filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])]
                
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    min_date, max_date = df[col].min(), df[col].max()
                    date_range = st.date_input(
                        f"Filter {human_name}",
                        [min_date, max_date],
                        key=f"date_{col}"
                    )
                    if len(date_range) == 2:
                        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                        filtered_df = filtered_df[(filtered_df[col] >= start) & (filtered_df[col] <= end)]

# Update the main DataFrame to the filtered version
df = filtered_df

# ===== STREAMLIT UI (UPDATED) =====
st.title("QuantPal AUS200 Dashboard")

## Section 1: Key Metrics and Data Insights
st.header("Key Insights")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Days", df.shape[0])
with col2:
    st.metric("Mean Daily Change (%)", f"{df['change_pct'].mean()}")
with col3:
    st.metric("Avg Open Gap (%)", f"{df['aus_gap_pct'].mean():.2f}")
with col4:
    st.metric("Avg RTH Range (pts)", f"{df['rth_range'].mean():.2f}")

# Histogram for Daily Change
st.subheader("Distribution of Daily Change (%)")
fig_hist = px.histogram(df, x='change_pct', nbins=50, title='Daily Change Distribution')
st.plotly_chart(fig_hist, use_container_width=True)

# Call the new function to plot the RTH high/low occurrences
st.header("RTH High and Low Time Occurrences")
plot_high_low_separate_charts(df)

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

        # Create a list of the unique dates to iterate over
        sorted_dates = st.session_state['matched_dates']

        # Loop through the dates in steps of 2
        for i in range(0, len(sorted_dates), 2):
            # Create two columns for each pair of charts
            col1, col2 = st.columns(2)

            # Get the data for the first chart in the pair
            date1_str = sorted_dates[i]
            daily_data1 = df_candles[df_candles['time'].dt.date.astype(str) == date1_str]
            
            with col1:
                if not daily_data1.empty:
                    fig1 = go.Figure(data=[go.Candlestick(
                        x=daily_data1['time'],
                        open=daily_data1['open'],
                        high=daily_data1['high'],
                        low=daily_data1['low'],
                        close=daily_data1['close'],
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    )])
                    fig1.update_layout(
                        title=f"AUS200 {TF} Intraday: {date1_str} (9AM-4PM NSW)",
                        xaxis_rangeslider_visible=False
                    )
                    st.plotly_chart(fig1, use_container_width=True)

            # Check if a second chart exists in the pair
            if i + 1 < len(sorted_dates):
                # Get the data for the second chart
                date2_str = sorted_dates[i+1]
                daily_data2 = df_candles[df_candles['time'].dt.date.astype(str) == date2_str]
                
                with col2:
                    if not daily_data2.empty:
                        fig2 = go.Figure(data=[go.Candlestick(
                            x=daily_data2['time'],
                            open=daily_data2['open'],
                            high=daily_data2['high'],
                            low=daily_data2['low'],
                            close=daily_data2['close'],
                            increasing_line_color='green',
                            decreasing_line_color='red'
                        )])
                        fig2.update_layout(
                            title=f"AUS200 {TF} Intraday: {date2_str} (9AM-4PM NSW)",
                            xaxis_rangeslider_visible=False
                        )
                        st.plotly_chart(fig2, use_container_width=True)

        st.success("All charts generated!")

con.close()