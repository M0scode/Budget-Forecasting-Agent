import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
import io

# Streamlit page configuration
st.set_page_config(page_title="Budget Forecasting Agent", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button { background-color: #2c3e50; color: white; }
    .stSelectbox { margin-bottom: 20px; }
    .title { color: #2c3e50; font-size: 2.5em; text-align: center; }
    .subtitle { color: #34495e; font-size: 1.2em; text-align: center; }
    .logo { color: #2980b9; font-size: 1.5em; font-weight: bold; text-align: center; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# Placeholder logo
st.markdown('<div class="logo">2S0 Technologies - Budget Forecasting</div>', unsafe_allow_html=True)

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

# Step 2: Load Data (CSV or Synthetic)
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['date', 'department', 'category', 'amount']
            if not all(col in df.columns for col in required_columns):
                st.error("CSV must contain columns: date, department, category, amount")
                return None
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None
    else:
        return generate_synthetic_data()

# Step 3: Preprocess Data
def preprocess_data(df, department, category):
    df = df[(df['department'] == department) & (df['category'] == category)]
    df['date'] = pd.to_datetime(df['date'])
    df_agg = df.groupby('date').sum(numeric_only=True).reset_index()
    df_prophet = df_agg[['date', 'amount']].rename(columns={'date': 'ds', 'amount': 'y'})
    return df_prophet

# Step 4: Train Forecasting Model
def train_forecast_model(df_prophet):
    if df_prophet.empty:
        return None, None
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)
    return model, forecast

# Step 5: Variance Analysis
def analyze_variances(df_prophet, forecast):
    if df_prophet.empty or forecast is None:
        return None, None
    actuals = df_prophet.tail(12).copy()
    forecast_subset = forecast[['ds', 'yhat']].tail(12)
    merged = actuals.merge(forecast_subset, on='ds')
    merged['variance'] = (merged['y'] - merged['yhat']) / merged['yhat'] * 100
    anomalies = merged[abs(merged['variance']) > 10].copy()
    return merged, anomalies

# Step 6: Plot Forecast
def plot_forecast(df_prophet, forecast):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_prophet['ds'], df_prophet['y'], label='Actual Expenses', color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Expenses', color='orange', linestyle='--')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                    color='orange', alpha=0.1, label='Confidence Interval')
    ax.set_title('Actual vs Forecasted Expenses')
    ax.set_xlabel('Date')
    ax.set_ylabel('Expenses ($)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

# Step 7: Generate Report
def generate_report(merged, anomalies, department, category):
    report = f"**Budget Forecasting Report - {datetime.now().strftime('%Y-%m-%d')}**\n"
    report += "=" * 50 + "\n"
    report += f"**Department**: {department}\n"
    report += f"**Category**: {category}\n"
    report += "**Summary**\n"
    report += f"- Total Forecasted Months: {len(merged) if merged is not None else 0}\n"
    report += f"- Anomalies Detected (Variance > 10%): {len(anomalies) if anomalies is not None else 0}\n\n"
    
    if anomalies is not None and not anomalies.empty:
        report += "**Anomalies**\n"
        for _, row in anomalies.iterrows():
            report += (f"- Date: {row['ds'].strftime('%Y-%m')}, Actual: ${row['y']:.2f}, "
                      f"Forecasted: ${row['yhat']:.2f}, Variance: {row['variance']:.2f}%\n")
    
    return report

# Streamlit App
def main():
    st.markdown('<div class="title">Budget Forecasting Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Interactive expense forecasting for financial planning</div>', unsafe_allow_html=True)
    
    # File uploader
    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (columns: date, department, category, amount)", type="csv")
    
    # Load data
    df = load_data(uploaded_file)
    if df is None:
        st.stop()
    
    # Filters
    st.sidebar.header("Filters")
    departments = df['department'].unique().tolist()
    department = st.sidebar.selectbox("Select Department", departments, index=0)
    categories = df[df['department'] == department]['category'].unique().tolist()
    category = st.sidebar.selectbox("Select Category", categories, index=0)
    
    # Button to trigger forecasting
    if st.button("Run Forecast"):
        with st.spinner("Generating forecast..."):
            # Preprocess data with filters
            df_prophet = preprocess_data(df, department, category)
            
            if df_prophet.empty:
                st.error(f"No data available for {department} - {category}")
                st.stop()
            
            # Train model and forecast
            model, forecast = train_forecast_model(df_prophet)
            if model is None:
                st.error("Failed to generate forecast")
                st.stop()
            
            # Analyze variances
            merged, anomalies = analyze_variances(df_prophet, forecast)
            
            # Display plot
            st.subheader(f"Forecast Visualization: {department} - {category}")
            fig = plot_forecast(df_prophet, forecast)
            st.pyplot(fig)
            
            # Download plot
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.download_button(
                label="Download Plot",
                data=buf,
                file_name=f"forecast_{department}_{category}.png",
                mime="image/png"
            )
            
            # Display report
            st.subheader("Forecast Report")
            report = generate_report(merged, anomalies, department, category)
            st.markdown(report)
            
            # Download report
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"forecast_report_{department}_{category}.txt",
                mime="text/plain"
            )
            
            # Display anomalies table
            if anomalies is not None and not anomalies.empty:
                st.subheader("Anomalies Table")
                anomalies_display = anomalies[['ds', 'y', 'yhat', 'variance']].copy()
                anomalies_display.columns = ['Date', 'Actual ($)', 'Forecasted ($)', 'Variance (%)']
                anomalies_display['Date'] = anomalies_display['Date'].dt.strftime('%Y-%m')
                st.dataframe(anomalies_display)
            else:
                st.write("No anomalies detected (variance > 10%).")

if __name__ == "__main__":
    main()