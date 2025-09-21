#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Aquaculture Health Predictor",
    page_icon="🐟",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
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
    }
    .stProgress > div > div > div > div {
        background-color: #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# Set display options
pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')

@st.cache_data
def load_and_process_data():
    """Load and process the aquaculture dataset"""
    # This would be replaced with your actual dataset loading code
    # For now, I'll create a sample dataset structure
    st.info("Loading and processing dataset...")
    
    # Create sample data (replace this with your actual dataset loading)
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'location': np.random.choice(['Kavaratti North', 'Kavaratti South', 'Agatti East', 
                                    'Agatti West', 'Minicoy Lagoon'], n_samples),
        'farm_type': np.random.choice(['Seaweed (Kappaphycus alvarezii)', 
                                     'Pearl Oyster (Pinctada fucata)',
                                     'Finfish (Grouper)',
                                     'Seaweed (Gracilaria edulis)',
                                     'Shrimp (Penaeus monodon)'], n_samples),
        'temperature': np.random.uniform(25, 32, n_samples),
        'salinity': np.random.uniform(33, 36, n_samples),
        'dissolved_oxygen': np.random.uniform(5, 8, n_samples),
        'ph': np.random.uniform(7.8, 8.4, n_samples),
        'nitrates': np.random.uniform(0.02, 0.25, n_samples),
        'phosphates': np.random.uniform(0.01, 0.08, n_samples),
        'ammonia': np.random.uniform(0.005, 0.05, n_samples),
        'chlorophyll': np.random.uniform(0.3, 1.5, n_samples),
        'turbidity': np.random.uniform(3, 15, n_samples),
        'microbiome_health_score': np.random.uniform(50, 95, n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create engineered features
    df['month'] = df['date'].dt.month
    df['temp_oxygen_interaction'] = df['temperature'] * df['dissolved_oxygen']
    df['nutrient_balance'] = df['nitrates'] / (df['phosphates'] + 0.001)
    df['stress_index'] = ((df['temperature'] - 28)**2 + (df['ph'] - 8.1)**2)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['total_nutrients'] = df['nitrates'] + df['phosphates']
    
    # Water quality index
    oxygen_component = (df['dissolved_oxygen'] / 7) * 0.3
    ph_component = (8.1 - abs(df['ph'] - 8.1)) * 0.3
    ammonia_component = (1 - np.minimum(df['ammonia'] / 0.03, 1)) * 0.2
    turbidity_component = (1 - np.minimum(df['turbidity'] / 10, 1)) * 0.2
    df['water_quality_index'] = oxygen_component + ph_component + ammonia_component + turbidity_component
    
    return df

@st.cache_resource
def train_model(_df):
    """Train the machine learning model"""
    st.info("Training machine learning model...")
    
    # Define features and target
    features = ['location', 'farm_type', 'temperature', 'salinity', 'dissolved_oxygen', 
                'ph', 'nitrates', 'phosphates', 'ammonia', 'chlorophyll', 'turbidity',
                'temp_oxygen_interaction', 'nutrient_balance', 'stress_index',
                'month_sin', 'month_cos', 'total_nutrients', 'water_quality_index']
    target = 'microbiome_health_score'
    
    X = _df[features]
    y = _df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify feature types
    categorical_features = ['location', 'farm_type']
    numeric_features = [col for col in features if col not in categorical_features]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    # Train Gradient Boosting model (best performing from your analysis)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42
        ))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    st.success(f"Model trained successfully! R² score: {r2:.3f}")
    
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
    df['nutrient_balance'] = df['nitrates'] / (df['phosphates'] + 0.001)
    df['stress_index'] = ((df['temperature'] - 28)**2 + (df['ph'] - 8.1)**2)
    df['total_nutrients'] = df['nitrates'] + df['phosphates']
    
    # Water quality index
    oxygen_component = (df['dissolved_oxygen'] / 7) * 0.3
    ph_component = (8.1 - abs(df['ph'] - 8.1)) * 0.3
    ammonia_component = (1 - np.minimum(df['ammonia'] / 0.03, 1)) * 0.2
    turbidity_component = (1 - np.minimum(df['turbidity'] / 10, 1)) * 0.2
    df['water_quality_index'] = oxygen_component + ph_component + ammonia_component + turbidity_component
    
    return df

def main():
    st.markdown('<h1 class="main-header">🐟 Aquaculture Health Predictor</h1>', unsafe_allow_html=True)
    
    # Load data and train model
    with st.spinner("Loading data and training model..."):
        df = load_and_process_data()
        model, features, preprocessor, X_test, y_test = train_model(df)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Single Prediction", "Batch Prediction", "Model Analysis", "Data Exploration"])
    
    with tab1:
        st.header("Single Farm Prediction")
        st.markdown("Enter the details below to predict the microbiome health score.")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="feature-input">', unsafe_allow_html=True)
                location = st.selectbox("Location", [
                    "Kavaratti North", "Kavaratti South", "Agatti East", 
                    "Agatti West", "Minicoy Lagoon"
                ])
                farm_type = st.selectbox("Farm Type", [
                    "Seaweed (Kappaphycus alvarezii)", 
                    "Pearl Oyster (Pinctada fucata)",
                    "Finfish (Grouper)",
                    "Seaweed (Gracilaria edulis)",
                    "Shrimp (Penaeus monodon)"
                ])
                temperature = st.slider("Temperature (°C)", 20.0, 35.0, 28.0, 0.1)
                salinity = st.slider("Salinity (ppt)", 30.0, 38.0, 34.5, 0.1)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="feature-input">', unsafe_allow_html=True)
                dissolved_oxygen = st.slider("Dissolved Oxygen (mg/L)", 3.0, 9.0, 6.0, 0.1)
                ph = st.slider("pH", 7.5, 8.5, 8.1, 0.01)
                nitrates = st.slider("Nitrates (mg/L)", 0.02, 0.3, 0.14, 0.001)
                phosphates = st.slider("Phosphates (mg/L)", 0.003, 0.09, 0.04, 0.001)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="feature-input">', unsafe_allow_html=True)
                ammonia = st.slider("Ammonia (mg/L)", 0.003, 0.08, 0.023, 0.001)
                chlorophyll = st.slider("Chlorophyll (μg/L)", 0.2, 1.9, 0.9, 0.01)
                turbidity = st.slider("Turbidity (NTU)", 2.4, 18.4, 5.8, 0.1)
                date = st.date_input("Date", datetime.now())
                st.markdown('</div>', unsafe_allow_html=True)
            
            submitted = st.form_submit_button("Predict Health Score", type="primary")
        
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
                'date': [date]
            })
            
            # Create engineered features
            input_data = create_engineered_features(input_data)
            
            # Make prediction
            try:
                prediction = model.predict(input_data[features])[0]
                
                # Display results
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Health Score", f"{prediction:.1f}")
                
                with col2:
                    if prediction >= 80:
                        st.success("✅ Excellent Health")
                    elif prediction >= 70:
                        st.info("🟢 Good Health")
                    elif prediction >= 60:
                        st.warning("🟡 Moderate Health")
                    else:
                        st.error("🔴 Poor Health - Needs Attention")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with tab2:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with multiple records for batch prediction.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
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
                    st.error(f"Missing columns in uploaded file: {missing_cols}")
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
                    
                    # Display results
                    st.success(f"Successfully processed {len(results_df)} records")
                    st.dataframe(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="aquaculture_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.header("Model Analysis")
        
        # Model performance
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("Mean Squared Error", f"{mse:.2f}")
        
        # Actual vs Predicted plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted Microbiome Health Score')
        st.pyplot(fig)
        
        # Feature importance (if available)
        if hasattr(model.named_steps['regressor'], 'feature_importances_'):
            st.subheader("Feature Importance")
            
            # Get feature names after preprocessing
            numeric_features = [col for col in features if col not in ['location', 'farm_type']]
            feature_names = numeric_features + list(model.named_steps['preprocessor']
                                                  .named_transformers_['cat']
                                                  .get_feature_names_out(['location', 'farm_type']))
            
            importances = model.named_steps['regressor'].feature_importances_
            feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
            ax.set_title('Top 10 Feature Importances')
            st.pyplot(fig)
    
    with tab4:
        st.header("Data Exploration")
        
        # Show dataset info
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Samples", df.shape[0])
        with col2:
            st.metric("Number of Features", df.shape[1])
        
        # Show basic statistics
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())
        
        # Health score distribution
        st.subheader("Health Score Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['microbiome_health_score'], kde=True, ax=ax)
        ax.set_title('Microbiome Health Score Distribution')
        st.pyplot(fig)
        
        # Health scores by farm type
        st.subheader("Health Scores by Farm Type")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x='farm_type', y='microbiome_health_score', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('Health Scores by Farm Type')
        st.pyplot(fig)

if __name__ == "__main__":
    main()