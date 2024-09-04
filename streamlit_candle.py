import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

def download_data(ticker, period="max"):
    data = yf.download(ticker, period=period)
    return data

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

def analyze_pinbars(data, lookback, wick_ratio):
    bearish_pinbars = []
    bullish_pinbars = []
    
    for i in range(lookback, len(data)):
        slice = data.iloc[i-lookback:i+1]
        candle = slice.iloc[-1]
        
        if is_uptrend(slice, lookback):
            if is_bearish_pinbar(candle, True, wick_ratio):
                bearish_pinbars.append(i)
        elif is_downtrend(slice, lookback):
            if is_bullish_pinbar(candle, True, wick_ratio):
                bullish_pinbars.append(i)
    
    return bearish_pinbars, bullish_pinbars

def calculate_success_rate(data, pinbars, multipliers, stop_loss_pct):
    results = {mult: {'success': 0, 'fail': 0} for mult in multipliers}
    
    for idx in pinbars:
        candle = data.iloc[idx]
        candle_size = candle['High'] - candle['Low']
        stop_loss = candle['High'] + (candle['Close'] * stop_loss_pct)
        
        for j in range(idx+1, len(data)):
            next_candle = data.iloc[j]
            
            if next_candle['High'] >= stop_loss:
                for mult in multipliers:
                    results[mult]['fail'] += 1
                break
            
            for mult in multipliers:
                target = candle['Close'] - (candle_size * mult)
                if next_candle['Low'] <= target:
                    results[mult]['success'] += 1
                    break
            else:
                continue
            break
    
    return results

def main():
    st.title("Price Pattern Analysis App")
    
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="0700.HK")
    lookback = st.sidebar.slider("Trend Defining Lookback Period", min_value=5, max_value=50, value=10)
    wick_ratio = st.sidebar.slider("Wick Ratio", min_value=0.5, max_value=0.95, value=0.75, step=0.05)
    stop_loss_pct = st.sidebar.slider("Stop Loss Percentage(% added to High of Trigger Bar", min_value=0.001, max_value=0.02, value=0.005, step=0.001)
    
    data = download_data(ticker)
    
    if data.empty:
        st.error(f"No data found for ticker {ticker}")
        return
    
    st.write(f"Analyzing {ticker} data...")

    # Display the period of the downloaded data
    st.write(f"Data period: from {data.index[0].date()} to {data.index[-1].date()} ({len(data)} trading days)")
    
    bearish_pinbars, bullish_pinbars = analyze_pinbars(data, lookback, wick_ratio)
    
        st.write(f"Total Bearish Pinbars: {len(bearish_pinbars)}")
    st.write(f"Total Bullish Pinbars: {len(bullish_pinbars)}")
    
    multipliers = [1, 1.5, 2, 3]
    
    bearish_results = calculate_success_rate(data, bearish_pinbars, multipliers, stop_loss_pct)
    bullish_results = calculate_success_rate(data, bullish_pinbars, multipliers, stop_loss_pct)
    
    st.subheader("Bearish Pinbar Results")
    for mult, result in bearish_results.items():
        total = result['success'] + result['fail']
        if total > 0:
            win_rate = result['success'] / total
            st.write(f"Target {mult}x candle size - Success: {result['success']}, Fail: {result['fail']}, Win Rate: {win_rate:.2%}")
    
    st.subheader("Bullish Pinbar Results")
    for mult, result in bullish_results.items():
        total = result['success'] + result['fail']
        if total > 0:
            win_rate = result['success'] / total
            st.write(f"Target {mult}x candle size - Success: {result['success']}, Fail: {result['fail']}, Win Rate: {win_rate:.2%}")

if __name__ == "__main__":
    main()
