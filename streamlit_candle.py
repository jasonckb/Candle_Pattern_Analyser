import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def download_data(ticker, period="max"):
    data = yf.download(ticker, period=period)
    return data

def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def is_uptrend(data, lookback):
    return data['High'].iloc[-1] > data['High'].iloc[-lookback-1]

def is_downtrend(data, lookback):
    return data['Low'].iloc[-1] < data['Low'].iloc[-lookback-1]

def is_bearish_pinbar(candle, uptrend, upper_wick_ratio):
    candle_size = candle['High'] - candle['Low']
    upper_wick = candle['High'] - max(candle['Open'], candle['Close'])
    return uptrend and (upper_wick / candle_size >= upper_wick_ratio)

def is_bullish_pinbar(candle, downtrend, lower_wick_ratio):
    candle_size = candle['High'] - candle['Low']
    lower_wick = min(candle['Open'], candle['Close']) - candle['Low']
    return downtrend and (lower_wick / candle_size >= lower_wick_ratio)

def is_bearish_engulfing(prev_candle, curr_candle, uptrend, body_ratio):
    prev_body_size = abs(prev_candle['Close'] - prev_candle['Open'])
    curr_body_size = abs(curr_candle['Close'] - curr_candle['Open'])
    prev_candle_size = prev_candle['High'] - prev_candle['Low']
    curr_candle_size = curr_candle['High'] - curr_candle['Low']
    
    return (uptrend and
            prev_candle['Close'] > prev_candle['Open'] and
            curr_candle['Close'] < curr_candle['Open'] and
            prev_body_size / prev_candle_size >= body_ratio and
            curr_body_size / curr_candle_size >= body_ratio and
            curr_body_size > prev_body_size)

def is_bullish_engulfing(prev_candle, curr_candle, downtrend, body_ratio):
    prev_body_size = abs(prev_candle['Close'] - prev_candle['Open'])
    curr_body_size = abs(curr_candle['Close'] - curr_candle['Open'])
    prev_candle_size = prev_candle['High'] - prev_candle['Low']
    curr_candle_size = curr_candle['High'] - curr_candle['Low']
    
    return (downtrend and
            prev_candle['Close'] < prev_candle['Open'] and
            curr_candle['Close'] > curr_candle['Open'] and
            prev_body_size / prev_candle_size >= body_ratio and
            curr_body_size / curr_candle_size >= body_ratio and
            curr_body_size > prev_body_size)

def analyze_patterns(data, lookback, wick_ratio, body_ratio):
    bearish_pinbars = []
    bullish_pinbars = []
    bearish_engulfing = []
    bullish_engulfing = []
    
    for i in range(lookback, len(data)):
        slice = data.iloc[i-lookback:i+1]
        candle = slice.iloc[-1]
        prev_candle = slice.iloc[-2]
        
        if is_uptrend(slice, lookback):
            if is_bearish_pinbar(candle, True, wick_ratio):
                bearish_pinbars.append(i)
            if is_bearish_engulfing(prev_candle, candle, True, body_ratio):
                bearish_engulfing.append(i)
        elif is_downtrend(slice, lookback):
            if is_bullish_pinbar(candle, True, wick_ratio):
                bullish_pinbars.append(i)
            if is_bullish_engulfing(prev_candle, candle, True, body_ratio):
                bullish_engulfing.append(i)
    
    return bearish_pinbars, bullish_pinbars, bearish_engulfing, bullish_engulfing

def calculate_success_rate(data, patterns, multipliers, atr, stop_loss_atr):
    results = {mult: {'success': 0, 'fail': 0} for mult in multipliers}
    mfe_list = []
    
    for idx in patterns:
        trigger_candle = data.iloc[idx]
        
        if trigger_candle['Close'] < trigger_candle['Open']:  # Bearish pattern
            stop_loss = trigger_candle['High'] + stop_loss_atr * atr[idx]
            risk = stop_loss - trigger_candle['Close']
            targets = [trigger_candle['Close'] - mult * risk for mult in multipliers]
            
            highest_reached = None
            max_favorable_excursion = 0
            for j in range(idx+1, min(idx+15, len(data))):
                next_candle = data.iloc[j]
                if next_candle['High'] >= stop_loss:
                    break
                favorable_excursion = (trigger_candle['Close'] - next_candle['Low']) / risk
                max_favorable_excursion = max(max_favorable_excursion, favorable_excursion)
                for i, target in enumerate(targets):
                    if next_candle['Low'] <= target:
                        highest_reached = i
            
            if max_favorable_excursion > 0:
                mfe_list.append(max_favorable_excursion)
            
            if highest_reached is not None:
                for i in range(highest_reached + 1):
                    results[multipliers[i]]['success'] += 1
                for i in range(highest_reached + 1, len(multipliers)):
                    results[multipliers[i]]['fail'] += 1
            else:
                for mult in multipliers:
                    results[mult]['fail'] += 1
                    
        else:  # Bullish pattern
            stop_loss = trigger_candle['Low'] - stop_loss_atr * atr[idx]
            risk = trigger_candle['Close'] - stop_loss
            targets = [trigger_candle['Close'] + mult * risk for mult in multipliers]
            
            highest_reached = None
            max_favorable_excursion = 0
            for j in range(idx+1, min(idx+15, len(data))):
                next_candle = data.iloc[j]
                if next_candle['Low'] <= stop_loss:
                    break
                favorable_excursion = (next_candle['High'] - trigger_candle['Close']) / risk
                max_favorable_excursion = max(max_favorable_excursion, favorable_excursion)
                for i, target in enumerate(targets):
                    if next_candle['High'] >= target:
                        highest_reached = i
            
            if max_favorable_excursion > 0:
                mfe_list.append(max_favorable_excursion)
            
            if highest_reached is not None:
                for i in range(highest_reached + 1):
                    results[multipliers[i]]['success'] += 1
                for i in range(highest_reached + 1, len(multipliers)):
                    results[multipliers[i]]['fail'] += 1
            else:
                for mult in multipliers:
                    results[mult]['fail'] += 1
    
    return results, mfe_list


def display_results(results, pattern_name):
    st.subheader(f"{pattern_name} Results")
    for mult, result in results.items():
        total = result['success'] + result['fail']
        if total > 0:
            win_rate = result['success'] / total
            st.write(f"Target {mult}x RR - Success: {result['success']}, Fail: {result['fail']}, Win Rate: {win_rate:.2%}")

def display_mfe_stats(mfe_list):
    if mfe_list:
        non_zero_mfe = [mfe for mfe in mfe_list if mfe > 0]
        if non_zero_mfe:
            min_mfe = min(non_zero_mfe)
            mean_mfe = sum(mfe_list) / len(mfe_list)
            median_mfe = sorted(mfe_list)[len(mfe_list) // 2]
            max_mfe = max(mfe_list)
            
            st.write("MFE Statistics (in R multiples, where 1R = initial risk):")
            st.write(f"Min MFE (excluding 0): {min_mfe:.2f}R")
            st.write(f"Mean MFE: {mean_mfe:.2f}R")
            st.write(f"Median MFE: {median_mfe:.2f}R")
            st.write(f"Max MFE: {max_mfe:.2f}R")
        else:
            st.write("All MFE values are 0. No meaningful MFE statistics available.")
    else:
        st.write("No MFE data available.")


def create_candlestick_chart(data, patterns, atr, stop_loss_atr, multipliers):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=('Candlestick Chart'))

    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='dodgerblue',
        decreasing_line_color='pink',
        name='Candlesticks'
    ))

    # Highlight patterns
    pattern_colors = {
        'Bearish Pinbar': 'yellow',
        'Bullish Pinbar': 'yellow',
        'Bearish Engulfing': 'orange',
        'Bullish Engulfing': 'orange'
    }

    for pattern_name, pattern_indices in patterns.items():
        for idx in pattern_indices:
            candle = data.iloc[idx]
            fig.add_trace(go.Candlestick(
                x=[candle.name],
                open=[candle['Open']],
                high=[candle['High']],
                low=[candle['Low']],
                close=[candle['Close']],
                increasing_line_color=pattern_colors[pattern_name],
                decreasing_line_color=pattern_colors[pattern_name],
                name=pattern_name,
                showlegend=False,
                hovertext=[pattern_name]
            ))

            # Add stop loss and target levels
            if candle['Close'] < candle['Open']:  # Bearish pattern
                stop_loss = candle['High'] + stop_loss_atr * atr[idx]
                risk = stop_loss - candle['Close']
                targets = [candle['Close'] - mult * risk for mult in multipliers]
            else:  # Bullish pattern
                stop_loss = candle['Low'] - stop_loss_atr * atr[idx]
                risk = candle['Close'] - stop_loss
                targets = [candle['Close'] + mult * risk for mult in multipliers]

            # Add stop loss line
            fig.add_trace(go.Scatter(
                x=[candle.name, candle.name + timedelta(days=5)],
                y=[stop_loss, stop_loss],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Stop Loss',
                showlegend=False
            ))

            # Add target lines
            for i, target in enumerate(targets):
                fig.add_trace(go.Scatter(
                    x=[candle.name, candle.name + timedelta(days=5)],
                    y=[target, target],
                    mode='lines',
                    line=dict(color='green', dash='dash'),
                    name=f'Target {multipliers[i]}x',
                    showlegend=False
                ))

    fig.update_layout(
        title='One Year Candlestick Chart with Pattern Analysis',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    return fig

def main():
    st.set_page_config(layout="wide")
    st.title("Price Pattern Analysis by Jason Chan")
    
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="0700.HK")
    
    st.sidebar.header("Pinbar Analysis")
    pinbar_lookback = st.sidebar.slider("Pinbar Up/Down Trend Defining Lookback Period", min_value=5, max_value=50, value=10, key="pinbar_lookback")
    wick_ratio = st.sidebar.slider("Wick Ratio", min_value=0.5, max_value=0.95, value=0.75, step=0.05)
    pinbar_stop_loss = st.sidebar.slider("Pinbar Stop Loss (ATR multiplier)", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="pinbar_stop_loss")
    
    st.sidebar.header("Engulfing Pattern Analysis")
    engulfing_lookback = st.sidebar.slider("Engulfing Up/Down Trend Defining Lookback Period", min_value=5, max_value=50, value=20, key="engulfing_lookback")
    body_ratio = st.sidebar.slider("Body Ratio", min_value=0.5, max_value=0.95, value=0.8, step=0.05)
    engulfing_stop_loss = st.sidebar.slider("Engulfing Stop Loss (ATR multiplier)", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="engulfing_stop_loss")
    
    # Download one year of data for the chart
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    chart_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Download all available data for the full analysis
    full_data = download_data(ticker)
    
    if full_data.empty:
        st.error(f"No data found for ticker {ticker}")
        return
    
    st.write(f"Analyzing {ticker} data...")
    st.write(f"Full data period: from {full_data.index[0].date()} to {full_data.index[-1].date()} ({len(full_data)} trading days)")
    
    st.info(f"Stop Loss Calculation: For bearish patterns, stop loss is set at the high of the trigger candle plus {pinbar_stop_loss:.1f} ATR for Pinbars and {engulfing_stop_loss:.1f} ATR for Engulfing patterns. For bullish patterns, it's set at the low of the trigger candle minus the same ATR multiplier. ATR is calculated over a 14-day period.")
    
    # Calculate ATR for both datasets
    chart_atr = calculate_atr(chart_data)
    full_atr = calculate_atr(full_data)
    
    # Analyze patterns for both datasets
    chart_bearish_pinbars, chart_bullish_pinbars, _, _ = analyze_patterns(chart_data, pinbar_lookback, wick_ratio, body_ratio)
    _, _, chart_bearish_engulfing, chart_bullish_engulfing = analyze_patterns(chart_data, engulfing_lookback, wick_ratio, body_ratio)
    
    full_bearish_pinbars, full_bullish_pinbars, _, _ = analyze_patterns(full_data, pinbar_lookback, wick_ratio, body_ratio)
    _, _, full_bearish_engulfing, full_bullish_engulfing = analyze_patterns(full_data, engulfing_lookback, wick_ratio, body_ratio)
    
    chart_patterns = {
        'Bearish Pinbar': chart_bearish_pinbars,
        'Bullish Pinbar': chart_bullish_pinbars,
        'Bearish Engulfing': chart_bearish_engulfing,
        'Bullish Engulfing': chart_bullish_engulfing
    }
    
    multipliers = [0.5, 1, 1.5, 2, 3]
    
    # Create and display the candlestick chart
    fig = create_candlestick_chart(chart_data, chart_patterns, chart_atr, max(pinbar_stop_loss, engulfing_stop_loss), multipliers)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display pattern counts for the charted period
    st.subheader("Pattern Counts (Last Year)")
    for pattern_name, pattern_indices in chart_patterns.items():
        st.write(f"{pattern_name}: {len(pattern_indices)}")
    
    # Display full analysis results
    st.header("Full Data Analysis")
    
    st.subheader("Pinbar Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Total Bearish Pinbars: {len(full_bearish_pinbars)}")
        bearish_pinbar_results, bearish_pinbar_mfe = calculate_success_rate(full_data, full_bearish_pinbars, multipliers, full_atr, pinbar_stop_loss)
        display_results(bearish_pinbar_results, f"Bearish Pinbar (Wick size = {wick_ratio:.0%} of Bar)")
        display_mfe_stats(bearish_pinbar_mfe)
    
    with col2:
        st.write(f"Total Bullish Pinbars: {len(full_bullish_pinbars)}")
        bullish_pinbar_results, bullish_pinbar_mfe = calculate_success_rate(full_data, full_bullish_pinbars, multipliers, full_atr, pinbar_stop_loss)
        display_results(bullish_pinbar_results, f"Bullish Pinbar (Wick size = {wick_ratio:.0%} of Bar)")
        display_mfe_stats(bullish_pinbar_mfe)
    
    st.subheader("Engulfing Pattern Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        st.write(f"Total Bearish Engulfing: {len(full_bearish_engulfing)}")
        bearish_engulfing_results, bearish_engulfing_mfe = calculate_success_rate(full_data, full_bearish_engulfing, multipliers, full_atr, engulfing_stop_loss)
        display_results(bearish_engulfing_results, f"Bearish Engulfing (Body size = {body_ratio:.0%} of Bar)")
        display_mfe_stats(bearish_engulfing_mfe)
    
    with col4:
        st.write(f"Total Bullish Engulfing: {len(full_bullish_engulfing)}")
        bullish_engulfing_results, bullish_engulfing_mfe = calculate_success_rate(full_data, full_bullish_engulfing, multipliers, full_atr, engulfing_stop_loss)
        display_results(bullish_engulfing_results, f"Bullish Engulfing (Body size = {body_ratio:.0%} of Bar)")
        display_mfe_stats(bullish_engulfing_mfe)

if __name__ == "__main__":
    main()




