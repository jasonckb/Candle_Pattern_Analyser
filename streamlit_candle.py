import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

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

def display_mfe_stats(mfe_list):
    if mfe_list:
        min_mfe = min(mfe_list)
        mean_mfe = sum(mfe_list) / len(mfe_list)
        median_mfe = sorted(mfe_list)[len(mfe_list) // 2]
        max_mfe = max(mfe_list)
        
        st.write("MFE Statistics:")
        st.write(f"Min MFE: {min_mfe:.2f}")
        st.write(f"Mean MFE: {mean_mfe:.2f}")
        st.write(f"Median MFE: {median_mfe:.2f}")
        st.write(f"Max MFE: {max_mfe:.2f}")
    else:
        st.write("No MFE data available.")



def display_results(results, pattern_name):
    st.subheader(f"{pattern_name} Results")
    for mult, result in results.items():
        total = result['success'] + result['fail']
        if total > 0:
            win_rate = result['success'] / total
            st.write(f"Target {mult}x RR - Success: {result['success']}, Fail: {result['fail']}, Win Rate: {win_rate:.2%}")

def main():
    st.set_page_config(layout="wide")
    st.title("Price Pattern Analysis by Jason Chan")
    
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="0700.HK")
    
    st.header("Pinbar Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Total Bearish Pinbars: {len(bearish_pinbars)}")
        bearish_pinbar_results, bearish_pinbar_mfe = calculate_success_rate(data, bearish_pinbars, multipliers, atr, pinbar_stop_loss)
        display_results(bearish_pinbar_results, f"Bearish Pinbar (Wick size = {wick_ratio:.0%} of Bar)")
        display_mfe_stats(bearish_pinbar_mfe)
    
    with col2:
        st.write(f"Total Bullish Pinbars: {len(bullish_pinbars)}")
        bullish_pinbar_results, bullish_pinbar_mfe = calculate_success_rate(data, bullish_pinbars, multipliers, atr, pinbar_stop_loss)
        display_results(bullish_pinbar_results, f"Bullish Pinbar (Wick size = {wick_ratio:.0%} of Bar)")
        display_mfe_stats(bullish_pinbar_mfe)
    
    st.header("Engulfing Pattern Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        st.write(f"Total Bearish Engulfing: {len(bearish_engulfing)}")
        bearish_engulfing_results, bearish_engulfing_mfe = calculate_success_rate(data, bearish_engulfing, multipliers, atr, engulfing_stop_loss)
        display_results(bearish_engulfing_results, f"Bearish Engulfing (Body size = {body_ratio:.0%} of Bar)")
        display_mfe_stats(bearish_engulfing_mfe)
    
    with col4:
        st.write(f"Total Bullish Engulfing: {len(bullish_engulfing)}")
        bullish_engulfing_results, bullish_engulfing_mfe = calculate_success_rate(data, bullish_engulfing, multipliers, atr, engulfing_stop_loss)
        display_results(bullish_engulfing_results, f"Bullish Engulfing (Body size = {body_ratio:.0%} of Bar)")
        display_mfe_stats(bullish_engulfing_mfe)

if __name__ == "__main__":
    main()
