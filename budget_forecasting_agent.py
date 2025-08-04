import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime

# Streamlit page configuration
st.set_page_config(page_title="Budget Forecasting Agent", layout="wide")

# Step 1: Generate Synthetic Data
def generate_synthetic_data():
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='M')
    categories = ['Salaries', 'Travel', 'Supplies']
    data = []
    
    for date in dates:
        for category in categories:
            base = 50000 if category == 'Salaries' else 10000 if category == 'Travel' else 5000
            seasonal = 1000 * np.sin((date.month - 1) * np.pi / 6)
            noise = np.random.normal(0, 500)
            amount = max(base + seasonal + noise, 1000)
            data.append([date, 'Finance', category, round(amount, 2)])
    
    df = pd.DataFrame(data, columns=['date', 'department', 'category', 'amount'])
    return df

# Step 2: Preprocess Data
def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df_agg = df.groupby('date').sum(numeric_only=True).reset_index()
    df_prophet = df_agg[['date', 'amount']].rename(columns={'date': 'ds', 'amount': 'y'})
    return df_prophet

# Step 3: Train Forecasting Model
def train_forecast_model(df_prophet):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)
    return model, forecast

# Step 4: Variance Analysis
def analyze_variances(df_prophet, forecast):
    actuals = df_prophet.tail(12).copy()
    forecast_subset = forecast[['ds', 'yhat']].tail(12)
    merged = actuals.merge(forecast_subset, on='ds')
    merged['variance'] = (merged['y'] - merged['yhat']) / merged['yhat'] * 100
    anomalies = merged[abs(merged['variance']) > 10].copy()
    return merged, anomalies

# Step 5: Plot Forecast
def plot_forecast(df_prophet, forecast):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_prophet['ds'], df_prophet['y'], label='Actual Expenses', color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Expenses', color='orange', linestyle='--')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                    color='orange', alpha=0.1, label='Confidence Interval')
    ax.set_title('Actual vs Forecasted Expenses (Finance Department)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Expenses ($)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

# Step 6: Generate Report
def generate_report(merged, anomalies):
    report = f"**Budget Forecasting Report - {datetime.now().strftime('%Y-%m-%d')}**\n"
    report += "=" * 50 + "\n"
    report += "**Summary**\n"
    report += f"- Total Forecasted Months: {len(merged)}\n"
    report += f"- Anomalies Detected (Variance > 10%): {len(anomalies)}\n\n"
    
    if not anomalies.empty:
        report += "**Anomalies**\n"
        for _, row in anomalies.iterrows():
            report += (f"- Date: {row['ds'].strftime('%Y-%m')}, Actual: ${row['y']:.2f}, "
                      f"Forecasted: ${row['yhat']:.2f}, Variance: {row['variance']:.2f}%\n")
    
    return report

# Streamlit App
def main():
    st.title("Budget Forecasting Agent")
    st.write("This app forecasts monthly expenses for the Finance department and detects significant variances.")
    
    # Button to trigger forecasting
    if st.button("Run Forecast"):
        with st.spinner("Generating forecast..."):
            # Generate and preprocess data
            df = generate_synthetic_data()
            df_prophet = preprocess_data(df)
            
            # Train model and forecast
            model, forecast = train_forecast_model(df_prophet)
            
            # Analyze variances
            merged, anomalies = analyze_variances(df_prophet, forecast)
            
            # Display plot
            st.subheader("Forecast Visualization")
            fig = plot_forecast(df_prophet, forecast)
            st.pyplot(fig)
            
            # Display report
            st.subheader("Forecast Report")
            report = generate_report(merged, anomalies)
            st.markdown(report)
            
            # Display anomalies table
            if not anomalies.empty:
                st.subheader("Anomalies Table")
                anomalies_display = anomalies[['ds', 'y', 'yhat', 'variance']].copy()
                anomalies_display.columns = ['Date', 'Actual ($)', 'Forecasted ($)', 'Variance (%)']
                anomalies_display['Date'] = anomalies_display['Date'].dt.strftime('%Y-%m')
                st.dataframe(anomalies_display)
            else:
                st.write("No anomalies detected (variance > 10%).")

if __name__ == "__main__":
    main()
