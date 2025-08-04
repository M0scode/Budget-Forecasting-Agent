import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime

# Step 1: Generate Synthetic Data
def generate_synthetic_data():
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='M')
    categories = ['Salaries', 'Travel', 'Supplies']
    data = []
    
    for date in dates:
        for category in categories:
            # Base amount with seasonality and noise
            base = 50000 if category == 'Salaries' else 10000 if category == 'Travel' else 5000
            seasonal = 1000 * np.sin((date.month - 1) * np.pi / 6)  # Seasonal fluctuation
            noise = np.random.normal(0, 500)  # Random noise
            amount = max(base + seasonal + noise, 1000)  # Ensure positive amounts
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

# Step 5: Visualize Results
def plot_forecast(df_prophet, forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(df_prophet['ds'], df_prophet['y'], label='Actual Expenses', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Expenses', color='orange', linestyle='--')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     color='orange', alpha=0.1, label='Confidence Interval')
    plt.title('Actual vs Forecasted Expenses (Finance Department)')
    plt.xlabel('Date')
    plt.ylabel('Expenses ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('forecast_plot.png')
    plt.close()

# Step 6: Generate Report
def generate_report(merged, anomalies):
    report = f"Budget Forecasting Report - {datetime.now().strftime('%Y-%m-%d')}\n"
    report += "=" * 50 + "\n"
    report += "Summary:\n"
    report += f"Total Forecasted Months: {len(merged)}\n"
    report += f"Anomalies Detected (Variance > 10%): {len(anomalies)}\n\n"
    
    if not anomalies.empty:
        report += "Anomalies:\n"
        for _, row in anomalies.iterrows():
            report += (f"Date: {row['ds'].strftime('%Y-%m')}, Actual: ${row['y']:.2f}, "
                      f"Forecasted: ${row['yhat']:.2f}, Variance: {row['variance']:.2f}%\n")
    
    return report

# Main Function
def main():
    # Generate and preprocess data
    df = generate_synthetic_data()
    df_prophet = preprocess_data(df)
    
    # Train model and forecast
    model, forecast = train_forecast_model(df_prophet)
    
    # Analyze variances
    merged, anomalies = analyze_variances(df_prophet, forecast)
    
    # Visualize results
    plot_forecast(df_prophet, forecast)
    
    # Generate report
    report = generate_report(merged, anomalies)
    with open('forecast_report.txt', 'w') as f:
        f.write(report)
    
    print("Forecasting complete. Check 'forecast_plot.png' and 'forecast_report.txt' for results.")

if __name__ == "__main__":
    main()
