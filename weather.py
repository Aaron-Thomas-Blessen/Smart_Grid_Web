import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

# Set Streamlit theme
st.set_page_config(page_title="Energy & Weather Dashboard", page_icon="⚡", layout="wide")

# Customize the style with Streamlit's theming options
st.markdown("""
    <style>
        .main {
            background-color: #f4f4f4;
        }
        h1, h2, h3, h4 {
            color: #4B7BEC;
        }
        .stButton button {
            background-color: #4B7BEC;
            color: white;
            border-radius: 8px;
        }
        .stFileUploader label {
            color: #4B7BEC;
        }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("⚡ Energy Consumption and Weather Data Dashboard")
st.write("A comprehensive dashboard to explore energy consumption and weather patterns. Upload your datasets to begin visualizing.")

# Sidebar for file upload
st.sidebar.title("Upload Data")
st.sidebar.write("Upload energy and weather datasets in CSV format to start visualizing.")
energy_file = st.sidebar.file_uploader("Upload Energy Dataset (CSV)", type="csv")
weather_file = st.sidebar.file_uploader("Upload Weather Dataset (CSV)", type="csv")

if energy_file and weather_file:
    # Load energy data
    df_energy = pd.read_csv(energy_file, parse_dates=['time'])
    df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True, infer_datetime_format=True)
    df_energy.set_index('time', inplace=True)

    # Load weather data
    df_weather = pd.read_csv(weather_file, parse_dates=['dt_iso'])
    df_weather['time'] = pd.to_datetime(df_weather['dt_iso'], utc=True, infer_datetime_format=True)
    df_weather.drop(['dt_iso'], axis=1, inplace=True)
    df_weather.set_index('time', inplace=True)

    # Display data preview
    st.header("📊 Data Previews")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Energy Dataset")
        st.write(df_energy.head())
    
    with col2:
        st.subheader("Weather Dataset")
        st.write(df_weather.head())

    # Data cleaning - Drop unnecessary columns
    df_energy = df_energy.drop(['generation fossil coal-derived gas', 'generation fossil oil shale',
                                'generation fossil peat', 'generation geothermal',
                                'generation hydro pumped storage aggregated', 'generation marine',
                                'generation wind offshore', 'forecast wind offshore eday ahead',
                                'total load forecast', 'forecast solar day ahead',
                                'forecast wind onshore day ahead'], axis=1)

    # Line plot of Actual Total Load for the first 2 weeks
    st.header("🔋 Energy Load Analysis")
    st.subheader("Actual Total Load (First 2 Weeks - Original)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_energy['total load actual'].iloc[:24 * 7 * 2], color='#4B7BEC')
    ax.set_title("Actual Total Load (First 2 Weeks)", fontsize=16)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Total Load (MWh)", fontsize=12)
    st.pyplot(fig)

    # Handling missing values
    st.header("🛠️ Data Cleaning")
    st.subheader("Handling Missing Values")
    missing_values_before = df_energy.isnull().sum().sum()
    st.write(f"Number of missing values before interpolation: {missing_values_before}")

    df_energy.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)
    missing_values_after = df_energy.isnull().sum().sum()
    st.write(f"Number of missing values after interpolation: {missing_values_after}")

    # Correlation Heatmap
    st.header("🌡️ Correlation Heatmap")
    correlations = df_energy.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Feature Engineering: Hour, Weekday, Month
    st.header("⚙️ Feature Engineering")
    df_energy['hour'] = df_energy.index.hour
    df_energy['weekday'] = df_energy.index.weekday
    df_energy['month'] = df_energy.index.month
    st.write("New features (`hour`, `weekday`, `month`) added to the energy dataset:")
    st.write(df_energy.head())

    # Data Scaling
    st.header("📏 Data Scaling")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_energy[['total load actual']])
    st.write("Scaled `total load actual` data:")
    st.write(scaled_data[:10])

    # Decomposition of Electricity Price
    st.header("📉 Seasonal Decomposition")
    res = sm.tsa.seasonal_decompose(df_energy['price actual'], model='additive', period=24)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
    res.observed.plot(ax=ax1, title='Observed')
    res.trend.plot(ax=ax2, title='Trend')
    res.seasonal.plot(ax=ax3, title='Seasonal')
    res.resid.plot(ax=ax4, title='Residual')
    plt.tight_layout()
    st.pyplot(fig)

    # RMSE Calculation Example
    st.header("📊 RMSE Calculation")
    y_true = df_energy['price actual'].iloc[:100]
    y_pred = y_true + np.random.normal(0, 0.05, size=len(y_true))  # Dummy prediction for illustration
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    st.write(f"Calculated RMSE: {rmse:.3f}")

else:
    st.warning("Please upload both the Energy and Weather datasets to proceed.")
