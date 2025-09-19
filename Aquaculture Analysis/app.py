#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Aquaculture Health Predictor",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved visibility
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sidebar-header {
        font-size: 1.5rem;
        color: #2E86AB;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
    .feature-input {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        color: #333333;
    }
    .recommendation-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        color: #333333;
    }
    .warning-box {
        background-color: #ffebee;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #f44336;
        margin: 1rem 0;
        color: #333333;
    }
    .stProgress > div > div > div > div {
        background-color: #2E86AB;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        color: #333333;
    }
    .farm-type-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        color: #333333;
        font-weight: 500;
    }
    .dark-theme-text {
        color: #333333 !important;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2E86AB;
        margin: 1rem 0;
        border-bottom: 2px solid #2E86AB;
        padding-bottom: 0.5rem;
        font-weight: bold;
    }
    .parameter-range {
        background-color: #fff3e0;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff9800;
        font-weight: 500;
    }
    .optimal-range-header {
        font-size: 1.2rem;
        color: #2E86AB;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Set display options
pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')

# Define optimal parameter ranges for different farm types
OPTIMAL_RANGES = {
    "Seaweed (Kappaphycus alvarezii)": {
        "temperature": (25, 30),
        "salinity": (30, 35),
        "dissolved_oxygen": (5, 7),
        "ph": (7.8, 8.4),
        "nitrates": (0.1, 0.3),
        "phosphates": (0.02, 0.06),
        "ammonia": (0, 0.02),
        "turbidity": (5, 15)
    },
    "Pearl Oyster (Pinctada fucata)": {
        "temperature": (25, 29),
        "salinity": (33, 36),
        "dissolved_oxygen": (5.5, 7.5),
        "ph": (7.9, 8.3),
        "nitrates": (0.05, 0.2),
        "phosphates": (0.01, 0.05),
        "ammonia": (0, 0.01),
        "turbidity": (3, 10)
    },
    "Finfish (Group er)": {
        "temperature": (26, 30),
        "salinity": (32, 35),
        "dissolved_oxygen": (6, 8),
        "ph": (7.8, 8.2),
        "nitrates": (0.05, 0.15),
        "phosphates": (0.01, 0.04),
        "ammonia": (0, 0.005),
        "turbidity": (2, 8)
    },
    "Seaweed (Gracilaria edulis)": {
        "temperature": (24, 29),
        "salinity": (30, 35),
        "dissolved_oxygen": (5, 7),
        "ph": (7.8, 8.4),
        "nitrates": (0.1, 0.3),
        "phosphates": (0.02, 0.06),
        "ammonia": (0, 0.02),
        "turbidity": (5, 15)
    },
    "Shrimp (Penaeus monodon)": {
        "temperature": (28, 32),
        "salinity": (15, 25),
        "dissolved_oxygen": (5, 7),
        "ph": (7.5, 8.5),
        "nitrates": (0.05, 0.2),
        "phosphates": (0.01, 0.05),
        "ammonia": (0, 0.01),
        "turbidity": (5, 20)
    }
}

@st.cache_data
def load_and_process_data():
    """Load and process the aquaculture dataset with realistic patterns"""
    st.info("📊 Loading and processing dataset...")
    
    # Create more realistic data with patterns instead of pure randomness
    np.random.seed(42)
    n_samples = 1000  # Reduced sample size from 5000 to 1000
    
    # Base data structure - dates only up to 2025
    end_date = datetime(2025, 12, 31)
    start_date = end_date - timedelta(days=n_samples-1)
    
    sample_data = {
        'date': pd.date_range(start_date, end_date, freq='D'),
        'location': np.random.choice(['Kavaratti North', 'Kavaratti South', 'Agatti East', 
                                    'Agatti West', 'Minicoy Lagoon'], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'farm_type': np.random.choice(list(OPTIMAL_RANGES.keys()), n_samples, p=[0.3, 0.2, 0.2, 0.2, 0.1]),
    }
    
    df = pd.DataFrame(sample_data)
    
    # Generate realistic parameter values based on farm type and seasonality
    for idx, row in df.iterrows():
        farm_type = row['farm_type']
        optimal = OPTIMAL_RANGES[farm_type]
        month = row['date'].month
        
        # Seasonal variations (more extreme in summer months)
        season_factor = 1 + 0.2 * np.sin(2 * np.pi * month / 12) if month in [5, 6, 7, 8] else 1
        
        # Generate parameters with some correlation to optimal ranges
        # Temperature with seasonal variation
        temp_mid = np.mean(optimal['temperature'])
        df.loc[idx, 'temperature'] = np.random.normal(
            loc=temp_mid, 
            scale=1.5 * season_factor
        )
        
        # Salinity with rainfall effect
        salinity_mid = np.mean(optimal['salinity'])
        if month in [6, 7, 8, 9]:  # Monsoon months
            salinity_mid *= 0.9  # Rainfall dilutes salinity
        df.loc[idx, 'salinity'] = np.random.normal(
            loc=salinity_mid, 
            scale=1.2
        )
        
        # Dissolved oxygen
        do_mid = np.mean(optimal['dissolved_oxygen'])
        df.loc[idx, 'dissolved_oxygen'] = np.random.normal(
            loc=do_mid, 
            scale=0.8
        )
        
        # pH
        ph_mid = np.mean(optimal['ph'])
        df.loc[idx, 'ph'] = np.random.normal(
            loc=ph_mid, 
            scale=0.15
        )
        
        # Nutrients - higher in productive systems
        df.loc[idx, 'nitrates'] = np.random.gamma(
            shape=2, 
            scale=np.mean(optimal['nitrates']) / 2
        )
        
        df.loc[idx, 'phosphates'] = np.random.gamma(
            shape=2, 
            scale=np.mean(optimal['phosphates']) / 2
        )
        
        # Ammonia - lower is better
        df.loc[idx, 'ammonia'] = np.random.exponential(
            scale=np.mean(optimal['ammonia']) * 2
        )
        
        # Chlorophyll - indicator of productivity
        df.loc[idx, 'chlorophyll'] = np.random.gamma(shape=2, scale=0.3)
        
        # Turbidity - affected by rainfall
        turbidity_mid = np.mean(optimal['turbidity'])
        if month in [6, 7, 8, 9]:  # Monsoon months
            turbidity_mid *= 1.3  # Rainfall increases turbidity
        df.loc[idx, 'turbidity'] = np.random.gamma(
            shape=2, 
            scale=turbidity_mid / 2
        )
        
        # Add some rainfall effect (higher in monsoon months)
        if month in [6, 7, 8, 9]:
            df.loc[idx, 'rainfall'] = np.random.exponential(scale=15)
        else:
            df.loc[idx, 'rainfall'] = np.random.exponential(scale=3)
    
    # Calculate a realistic health score based on parameter deviations from optimal
    health_scores = []
    for idx, row in df.iterrows():
        farm_type = row['farm_type']
        optimal = OPTIMAL_RANGES[farm_type]
        
        score = 100  # Start with perfect score
        
        # Penalize deviations from optimal ranges
        for param, (low, high) in optimal.items():
            if param in row:
                value = row[param]
                # Calculate how far the value is from the optimal range
                if value < low:
                    deviation = (low - value) / (high - low)
                    penalty = min(30, deviation * 20)
                    score -= penalty
                elif value > high:
                    deviation = (value - high) / (high - low)
                    penalty = min(30, deviation * 20)
                    score -= penalty
        
        # Additional penalties for extreme values
        if row['ammonia'] > optimal['ammonia'][1] * 1.5:
            score -= 15
        if row['dissolved_oxygen'] < optimal['dissolved_oxygen'][0] * 0.8:
            score -= 20
            
        # Ensure score is within bounds
        score = max(20, min(98, score))
        
        # Add some random noise
        score += np.random.normal(0, 2)
        
        health_scores.append(score)
    
    df['microbiome_health_score'] = health_scores
    
    # Create engineered features
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    
    # Trigonometric encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Parameter interactions
    df['temp_oxygen_interaction'] = df['temperature'] * df['dissolved_oxygen']
    df['nutrient_ratio'] = df['nitrates'] / (df['phosphates'] + 0.001)
    df['stress_index'] = ((df['temperature'] - 28)*2 + (df['ph'] - 8.1)*2) / 10
    df['total_nutrients'] = df['nitrates'] + df['phosphates']
    
    # Water quality index calculation
    for idx, row in df.iterrows():
        farm_type = row['farm_type']
        optimal = OPTIMAL_RANGES[farm_type]
        
        # Calculate how close each parameter is to its optimal range (0-1 scale)
        oxygen_score = 1 - min(1, max(0, abs(row['dissolved_oxygen'] - np.mean(optimal['dissolved_oxygen'])) / (optimal['dissolved_oxygen'][1] - optimal['dissolved_oxygen'][0])))
        ph_score = 1 - min(1, max(0, abs(row['ph'] - np.mean(optimal['ph'])) / (optimal['ph'][1] - optimal['ph'][0])))
        ammonia_score = 1 - min(1, max(0, row['ammonia'] / (optimal['ammonia'][1] * 1.5)))
        turbidity_score = 1 - min(1, max(0, (row['turbidity'] - optimal['turbidity'][0]) / (optimal['turbidity'][1] - optimal['turbidity'][0])))
        
        # Weighted water quality index
        df.loc[idx, 'water_quality_index'] = (
            oxygen_score * 0.3 + 
            ph_score * 0.25 + 
            ammonia_score * 0.25 + 
            turbidity_score * 0.2
        )
    
    # Additional environmental factors
    df['productivity_index'] = df['chlorophyll'] * df['water_quality_index'] * 10
    df['stability_index'] = 1 / (1 + df['stress_index'])
    
    return df

@st.cache_resource
def train_model(_df):
    """Train the machine learning model with improved pipeline"""
    st.info("🤖 Training machine learning model...")
    
    # Define features and target
    features = ['location', 'farm_type', 'temperature', 'salinity', 'dissolved_oxygen', 
                'ph', 'nitrates', 'phosphates', 'ammonia', 'chlorophyll', 'turbidity',
                'rainfall', 'temp_oxygen_interaction', 'nutrient_ratio', 'stress_index',
                'month_sin', 'month_cos', 'total_nutrients', 'water_quality_index',
                'productivity_index', 'stability_index']
    
    target = 'microbiome_health_score'
    
    X = _df[features]
    y = _df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify feature types
    categorical_features = ['location', 'farm_type']
    numeric_features = [col for col in features if col not in categorical_features]
    
    # Create improved preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create model with hyperparameter tuning
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_regression, k='all')),  # Changed to 'all' to avoid index issues
        ('regressor', xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    st.success(f"✅ Model trained successfully! R² score: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    return model, features, preprocessor, X_test, y_test

def create_engineered_features(input_df):
    """Create engineered features from input data"""
    df = input_df.copy()
    
    # Create month from date if available
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Create other engineered features
    df['temp_oxygen_interaction'] = df['temperature'] * df['dissolved_oxygen']
    df['nutrient_ratio'] = df['nitrates'] / (df['phosphates'] + 0.001)
    df['stress_index'] = ((df['temperature'] - 28)*2 + (df['ph'] - 8.1)*2) / 10
    df['total_nutrients'] = df['nitrates'] + df['phosphates']
    
    # Calculate water quality index based on farm type
    for idx, row in df.iterrows():
        farm_type = row['farm_type']
        optimal = OPTIMAL_RANGES[farm_type]
        
        # Calculate how close each parameter is to its optimal range (0-1 scale)
        oxygen_score = 1 - min(1, max(0, abs(row['dissolved_oxygen'] - np.mean(optimal['dissolved_oxygen'])) / (optimal['dissolved_oxygen'][1] - optimal['dissolved_oxygen'][0])))
        ph_score = 1 - min(1, max(0, abs(row['ph'] - np.mean(optimal['ph'])) / (optimal['ph'][1] - optimal['ph'][0])))
        ammonia_score = 1 - min(1, max(0, row['ammonia'] / (optimal['ammonia'][1] * 1.5)))
        turbidity_score = 1 - min(1, max(0, (row['turbidity'] - optimal['turbidity'][0]) / (optimal['turbidity'][1] - optimal['turbidity'][0])))
        
        # Weighted water quality index
        df.loc[idx, 'water_quality_index'] = (
            oxygen_score * 0.3 + 
            ph_score * 0.25 + 
            ammonia_score * 0.25 + 
            turbidity_score * 0.2
        )
    
    # Additional indices
    df['productivity_index'] = df['chlorophyll'] * df['water_quality_index'] * 10
    df['stability_index'] = 1 / (1 + df['stress_index'])
    
    # Add rainfall if not present (default to average)
    if 'rainfall' not in df.columns:
        df['rainfall'] = 8.0  # Default average value
    
    return df

def generate_recommendations(input_data, prediction):
    """Generate recommendations based on input parameters and prediction"""
    recommendations = []
    warnings = []
    
    farm_type = input_data['farm_type'].iloc[0]
    optimal = OPTIMAL_RANGES[farm_type]
    
    # Check each parameter against optimal ranges
    for param, (low, high) in optimal.items():
        if param in input_data.columns:
            value = input_data[param].iloc[0]
            if value < low:
                recommendations.append(f"📈 Increase {param} (current: {value:.2f}, optimal: {low}-{high})")
                warnings.append(f"⚠ Low {param} can negatively impact {farm_type.split()[0]} health")
            elif value > high:
                recommendations.append(f"📉 Decrease {param} (current: {value:.2f}, optimal: {low}-{high})")
                warnings.append(f"⚠ High {param} can stress {farm_type.split()[0]} organisms")
    
    # Special cases for critical parameters
    if input_data['dissolved_oxygen'].iloc[0] < optimal['dissolved_oxygen'][0] * 0.9:
        recommendations.append("💨 Increase aeration or reduce stocking density to improve dissolved oxygen levels.")
    
    if input_data['ammonia'].iloc[0] > optimal['ammonia'][1] * 1.2:
        recommendations.append("⚠ Immediately reduce feeding rates and increase water exchange to control ammonia levels.")
        warnings.append("☠ CRITICAL: High ammonia levels are extremely toxic")
    
    # General recommendations based on health score
    if prediction < 60:
        recommendations.append("🔴 CRITICAL: Immediate action required. Consider consulting with aquaculture specialists.")
    elif prediction < 70:
        recommendations.append("🟡 Monitor closely and implement corrective measures for parameters outside optimal ranges.")
    elif prediction < 80:
        recommendations.append("🟢 Good conditions. Maintain current practices with regular monitoring.")
    else:
        recommendations.append("✅ Excellent conditions. Optimal for growth and production.")
    
    # Farm-type specific recommendations
    if "Seaweed" in farm_type:
        recommendations.append("🌿 For seaweed, ensure adequate water movement and light penetration.")
    elif "Oyster" in farm_type or "Shrimp" in farm_type:
        recommendations.append("🦪 Regularly clean nets and structures to prevent biofouling.")
    elif "Finfish" in farm_type:
        recommendations.append("🐟 Monitor feeding rates and adjust based on water temperature and fish activity.")
    
    return recommendations, warnings

def display_optimal_ranges(farm_type):
    """Display optimal parameter ranges for the selected farm type"""
    st.markdown(f'<div class="optimal-range-header">🎯 Optimal Ranges for {farm_type}</div>', unsafe_allow_html=True)
    
    optimal = OPTIMAL_RANGES[farm_type]
    for param, (low, high) in optimal.items():
        st.markdown(f'<div class="parameter-range"><b>{param.capitalize()}</b>: {low} - {high}</div>', unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🐟 Aquaculture Health Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📋 About</div>', unsafe_allow_html=True)
        st.info("🌊 This application predicts aquaculture health based on environmental parameters and provides recommendations for optimal farming conditions.")
        
        st.markdown('<div class="sidebar-header">🏭 Farm Types</div>', unsafe_allow_html=True)
        for farm_type in OPTIMAL_RANGES.keys():
            st.markdown(f'<div class="farm-type-card">{farm_type}</div>', unsafe_allow_html=True)
    
    # Load data and train model
    with st.spinner("📊 Loading data and training model..."):
        df = load_and_process_data()
        model, features, preprocessor, X_test, y_test = train_model(df)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Single Prediction", "📊 Batch Prediction", "📈 Model Analysis", "🔍 Data Exploration"])
    
    with tab1:
        st.markdown('<h2 class="section-header">🔮 Single Farm Prediction</h2>', unsafe_allow_html=True)
        st.markdown("📝 Enter the details below to predict the microbiome health score and get recommendations.")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="feature-input">', unsafe_allow_html=True)
                st.markdown("📍 Location**")
                location = st.selectbox("Select Location", [
                    "Kavaratti North", "Kavaratti South", "Agatti East", 
                    "Agatti West", "Minicoy Lagoon"
                ], label_visibility="collapsed")
                
                st.markdown("🏭 Farm Type**")
                farm_type = st.selectbox("Select Farm Type", list(OPTIMAL_RANGES.keys()), label_visibility="collapsed")
                
                # Display optimal ranges for selected farm type
                display_optimal_ranges(farm_type)
                
                st.markdown("🌡 Temperature (°C)")
                temperature = st.slider("Temperature", 20.0, 35.0, 28.0, 0.1, label_visibility="collapsed")
                
                st.markdown("🧂 Salinity (ppt)")
                salinity = st.slider("Salinity", 25.0, 40.0, 34.5, 0.1, label_visibility="collapsed")
                
                st.markdown("🌧 Rainfall (mm)")
                rainfall = st.slider("Rainfall", 0.0, 50.0, 8.0, 0.1, label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="feature-input">', unsafe_allow_html=True)
                st.markdown("💨 Dissolved Oxygen (mg/L)")
                dissolved_oxygen = st.slider("Dissolved Oxygen", 3.0, 10.0, 6.0, 0.1, label_visibility="collapsed")
                
                st.markdown("🧪 pH**")
                ph = st.slider("pH", 7.0, 9.0, 8.1, 0.01, label_visibility="collapsed")
                
                st.markdown("🌿 Nitrates (mg/L)")
                nitrates = st.slider("Nitrates", 0.01, 0.5, 0.14, 0.001, label_visibility="collapsed")
                
                st.markdown("🌱 Phosphates (mg/L)")
                phosphates = st.slider("Phosphates", 0.001, 0.1, 0.04, 0.001, label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="feature-input">', unsafe_allow_html=True)
                st.markdown("⚠ Ammonia (mg/L)")
                ammonia = st.slider("Ammonia", 0.001, 0.1, 0.023, 0.001, label_visibility="collapsed")
                
                st.markdown("🍃 Chlorophyll (μg/L)")
                chlorophyll = st.slider("Chlorophyll", 0.1, 3.0, 0.9, 0.01, label_visibility="collapsed")
                
                st.markdown("🌊 Turbidity (NTU)")
                turbidity = st.slider("Turbidity", 1.0, 30.0, 5.8, 0.1, label_visibility="collapsed")
                
                st.markdown("📅 Date**")
                date = st.date_input("Select Date", datetime.now(), label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
            
            submitted = st.form_submit_button("🔮 Predict Health Score", type="primary")
        
        if submitted:
            # Create input dataframe
            input_data = pd.DataFrame({
                'location': [location],
                'farm_type': [farm_type],
                'temperature': [temperature],
                'salinity': [salinity],
                'dissolved_oxygen': [dissolved_oxygen],
                'ph': [ph],
                'nitrates': [nitrates],
                'phosphates': [phosphates],
                'ammonia': [ammonia],
                'chlorophyll': [chlorophyll],
                'turbidity': [turbidity],
                'rainfall': [rainfall],
                'date': [date]
            })
            
            # Create engineered features
            input_data_processed = create_engineered_features(input_data)
            
            # Make prediction
            try:
                prediction = model.predict(input_data_processed[features])[0]
                
                # Display results
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown('<h3 class="dark-theme-text">📊 Prediction Results</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("🎯 Predicted Health Score", f"{prediction:.1f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("💧 Water Quality Index", f"{input_data_processed['water_quality_index'].iloc[0]*100:.1f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    if prediction >= 80:
                        st.success("✅ Excellent Health")
                    elif prediction >= 70:
                        st.info("🟢 Good Health")
                    elif prediction >= 60:
                        st.warning("🟡 Moderate Health")
                    else:
                        st.error("🔴 Poor Health")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Generate and display recommendations
                recommendations, warnings = generate_recommendations(input_data, prediction)
                
                if warnings:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown('<h3 class="dark-theme-text">⚠ Potential Issues</h3>', unsafe_allow_html=True)
                    for warning in warnings:
                        st.markdown(f"- {warning}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if recommendations:
                    st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                    st.markdown('<h3 class="dark-theme-text">📋 Recommendations</h3>', unsafe_allow_html=True)
                    for recommendation in recommendations:
                        st.markdown(f"- {recommendation}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
    
    with tab2:
        st.markdown('<h2 class="section-header">📊 Batch Prediction</h2>', unsafe_allow_html=True)
        st.markdown("📤 Upload a CSV file with multiple records for batch prediction.")
        
        # Download template
        template_data = {
            'location': ['Kavaratti North'],
            'farm_type': ['Seaweed (Kappaphycus alvarezii)'],
            'temperature': [28.0],
            'salinity': [34.5],
            'dissolved_oxygen': [6.0],
            'ph': [8.1],
            'nitrates': [0.14],
            'phosphates': [0.04],
            'ammonia': [0.023],
            'chlorophyll': [0.9],
            'turbidity': [5.8],
            'rainfall': [8.0],
            'date': [datetime.now().date()]
        }
        template_df = pd.DataFrame(template_data)
        csv = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Template CSV",
            data=csv,
            file_name="aquaculture_template.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader("📁 Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                batch_data = pd.read_csv(uploaded_file)
                
                # Check if required columns are present
                required_cols = ['location', 'farm_type', 'temperature', 'salinity', 
                               'dissolved_oxygen', 'ph', 'nitrates', 'phosphates', 
                               'ammonia', 'chlorophyll', 'turbidity']
                
                missing_cols = set(required_cols) - set(batch_data.columns)
                if missing_cols:
                    st.error(f"❌ Missing columns in uploaded file: {missing_cols}")
                else:
                    # Create engineered features
                    batch_data_processed = create_engineered_features(batch_data)
                    
                    # Make predictions
                    predictions = model.predict(batch_data_processed[features])
                    
                    # Add predictions to dataframe
                    results_df = batch_data.copy()
                    results_df['predicted_health_score'] = predictions
                    results_df['health_status'] = pd.cut(predictions, 
                                                       bins=[0, 60, 70, 80, 100],
                                                       labels=['Poor', 'Moderate', 'Good', 'Excellent'])
                    
                    # Calculate water quality index for each row
                    water_quality_indices = []
                    for idx, row in batch_data_processed.iterrows():
                        water_quality_indices.append(row['water_quality_index'] * 100)
                    results_df['water_quality_index'] = water_quality_indices
                    
                    # Display results
                    st.success(f"✅ Successfully processed {len(results_df)} records")
                    
                    # Show summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("📊 Average Score", f"{results_df['predicted_health_score'].mean():.1f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("🏆 Best Score", f"{results_df['predicted_health_score'].max():.1f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("📉 Worst Score", f"{results_df['predicted_health_score'].min():.1f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col4:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        poor_count = (results_df['health_status'] == 'Poor').sum()
                        st.metric("🔴 Poor Health", f"{poor_count}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show detailed results with pagination
                    st.dataframe(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="aquaculture_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    with tab3:
        st.markdown('<h2 class="section-header">📈 Model Analysis</h2>', unsafe_allow_html=True)
        
        # Model performance
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📊 R² Score", f"{r2:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📉 Mean Squared Error", f"{mse:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📏 RMSE", f"{rmse:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📐 MAE", f"{mae:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Cross-validation scores
        st.markdown('<h3 class="dark-theme-text">📊 Cross-Validation Results</h3>', unsafe_allow_html=True)
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='r2')
        col1, col2, col3, col4, col5 = st.columns(5)
        for i, score in enumerate(cv_scores):
            with [col1, col2, col3, col4, col5][i]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(f"Fold {i+1} R²", f"{score:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Actual vs Predicted plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('📈 Actual vs Predicted Microbiome Health Score')
        st.pyplot(fig)
        
        # Residual plot - FIXED THE TYPO HERE (changed figsize to figsize)
        fig, ax = plt.subplots(figsize=(8, 6))
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('📉 Residual Plot')
        st.pyplot(fig)
        
        # Feature importance (if available)
        try:
            st.markdown('<h3 class="dark-theme-text">⚖ Feature Importance</h3>', unsafe_allow_html=True)
            
            # For tree-based models
            if hasattr(model.named_steps['regressor'], 'feature_importances_'):
                # Get feature names after preprocessing
                numeric_features = [col for col in features if col not in ['location', 'farm_type']]
                
                # Get the feature names from the preprocessor
                preprocessor = model.named_steps['preprocessor']
                
                # Get numeric feature names after polynomial expansion
                numeric_transformer = preprocessor.named_transformers_['num']
                poly = numeric_transformer.named_steps['poly']
                numeric_feature_names = poly.get_feature_names_out(numeric_features)
                
                # Get categorical feature names
                categorical_transformer = preprocessor.named_transformers_['cat']
                onehot = categorical_transformer.named_steps['onehot']
                categorical_feature_names = onehot.get_feature_names_out(['location', 'farm_type'])
                
                # Combine all feature names
                all_feature_names = list(numeric_feature_names) + list(categorical_feature_names)
                
                # Get importances
                importances = model.named_steps['regressor'].feature_importances_
                
                # Create feature importance dataframe
                feature_importance = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
                feature_importance = feature_importance.sort_values('importance', ascending=False).head(15)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
                ax.set_title('📊 Top 15 Feature Importances')
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not display feature importances: {str(e)}")
    
    with tab4:
        st.markdown('<h2 class="section-header">🔍 Data Exploration</h2>', unsafe_allow_html=True)
        
        # Show dataset info
        st.markdown('<h3 class="dark-theme-text">📋 Dataset Overview</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📊 Total Samples", df.shape[0])
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📈 Number of Features", df.shape[1])
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📅 Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show basic statistics
        st.markdown('<h3 class="dark-theme-text">📊 Basic Statistics</h3>', unsafe_allow_html=True)
        st.dataframe(df.describe())
        
        # Health score distribution
        st.markdown('<h3 class="dark-theme-text">📈 Health Score Distribution</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['microbiome_health_score'], kde=True, ax=ax, bins=20)
        ax.set_title('📊 Microbiome Health Score Distribution')
        st.pyplot(fig)
        
        # Health scores by farm type
        st.markdown('<h3 class="dark-theme-text">🏭 Health Scores by Farm Type</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x='farm_type', y='microbiome_health_score', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('📊 Health Scores by Farm Type')
        st.pyplot(fig)
        
        # Time series of health scores
        st.markdown('<h3 class="dark-theme-text">📅 Health Score Trends Over Time</h3>', unsafe_allow_html=True)
        monthly_health = df.groupby(pd.Grouper(key='date', freq='M'))['microbiome_health_score'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(monthly_health['date'], monthly_health['microbiome_health_score'], marker='o')
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Health Score')
        ax.set_title('📅 Monthly Average Health Score Trend')
        ax.grid(True)
        st.pyplot(fig)
        
        # Correlation heatmap
        st.markdown('<h3 class="dark-theme-text">🔄 Correlation Heatmap</h3>', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax, 
                   cbar_kws={"shrink": .8}, square=True)
        ax.set_title('🔄 Correlation Matrix')
        st.pyplot(fig)
        
        # 3D Visualizations with Plotly
        st.markdown('<h3 class="dark-theme-text">📊 3D Visualizations</h3>', unsafe_allow_html=True)
        
        # Create tabs for different 3D visualizations
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["3D Health Parameters", "3D Time Analysis", "3D Farm Comparison"])
        
        with viz_tab1:
            st.markdown('<h4 class="dark-theme-text">🌡 3D Parameter Relationship</h4>', unsafe_allow_html=True)
            
            # Sample data for better performance in 3D visualization
            sample_df = df.sample(n=min(500, len(df)), random_state=42)
            
            # Create 3D scatter plot
            fig = px.scatter_3d(
                sample_df,
                x='temperature',
                y='dissolved_oxygen',
                z='ph',
                color='microbiome_health_score',
                color_continuous_scale='Viridis',
                title='3D Relationship: Temperature, Dissolved Oxygen, pH vs Health Score',
                labels={
                    'temperature': 'Temperature (°C)',
                    'dissolved_oxygen': 'Dissolved Oxygen (mg/L)',
                    'ph': 'pH',
                    'microbiome_health_score': 'Health Score'
                },
                hover_data=['farm_type', 'location']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            st.markdown('<h4 class="dark-theme-text">📅 3D Time Analysis</h4>', unsafe_allow_html=True)
            
            # Aggregate data by month and farm type
            monthly_data = df.groupby([pd.Grouper(key='date', freq='M'), 'farm_type']).agg({
                'microbiome_health_score': 'mean',
                'temperature': 'mean',
                'dissolved_oxygen': 'mean',
                'ph': 'mean'
            }).reset_index()
            
            fig = px.scatter_3d(
                monthly_data,
                x='date',
                y='temperature',
                z='microbiome_health_score',
                color='farm_type',
                title='3D Time Analysis: Date, Temperature vs Health Score by Farm Type',
                labels={
                    'date': 'Date',
                    'temperature': 'Temperature (°C)',
                    'microbiome_health_score': 'Health Score',
                    'farm_type': 'Farm Type'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab3:
            st.markdown('<h4 class="dark-theme-text">🏭 3D Farm Comparison</h4>', unsafe_allow_html=True)
            
            # Aggregate data by location and farm type
            location_data = df.groupby(['location', 'farm_type']).agg({
                'microbiome_health_score': 'mean',
                'temperature': 'mean',
                'dissolved_oxygen': 'mean',
                'ph': 'mean'
            }).reset_index()
            
            fig = px.scatter_3d(
                location_data,
                x='temperature',
                y='dissolved_oxygen',
                z='microbiome_health_score',
                color='location',
                symbol='farm_type',
                title='3D Farm Comparison: Temperature, Dissolved Oxygen vs Health Score',
                labels={
                    'temperature': 'Temperature (°C)',
                    'dissolved_oxygen': 'Dissolved Oxygen (mg/L)',
                    'microbiome_health_score': 'Health Score',
                    'location': 'Location',
                    'farm_type': 'Farm Type'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

if _name_ == "_main_":
    main()