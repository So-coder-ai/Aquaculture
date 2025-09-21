#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')


# Load your dataset
df = pd.read_csv('C:/Users/narai/Downloads/Aquaculture_Project/lakshadweep_aquaculture_data_2023_enhanced.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Display basic info
print(f"Dataset shape: {df.shape}")
df.head()


# In[6]:


# Basic information about the dataset
print("Data types:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# Summary statistics
df.describe()


# In[9]:


# Temperature trends
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='date', y='temperature', hue='location')
plt.title('Temperature Trends by Location')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Health score distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['microbiome_health_score'], kde=True)
plt.title('Microbiome Health Score Distribution')
plt.show()


# In[7]:


# Health scores by farm type
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='farm_type', y='microbiome_health_score')
plt.title('Health Scores by Farm Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Average parameters by farm type
farm_comparison = df.groupby('farm_type')[['temperature', 'dissolved_oxygen', 'ph', 'microbiome_health_score']].mean()
farm_comparison


# In[11]:


# Risk indicators over time
plt.figure(figsize=(14, 8))
df.set_index('date')[['disease_risk', 'bloom_risk', 'equipment_failure_risk']].plot()
plt.title('Risk Indicators Over Time')
plt.ylabel('Risk Score')
plt.tight_layout()
plt.show()

# High risk days
high_risk = df[df['disease_risk'] > 0.7]
print(f"Number of high disease risk days: {len(high_risk)}")
high_risk[['date', 'location', 'farm_type', 'disease_risk', 'bloom_risk']].head(10)


# In[8]:


# Import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline  # This line was missing!

# Select features for prediction
features = ['location', 'farm_type', 'temperature', 'salinity', 'dissolved_oxygen', 
            'ph', 'nitrates', 'phosphates', 'ammonia', 'chlorophyll', 'turbidity']
target = 'microbiome_health_score'

X = df[features]  # Feature matrix
y = df[target]    # Target vector

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify feature types
categorical_features = ['location', 'farm_type']
numeric_features = [col for col in features if col not in categorical_features]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),      # Scale numerical features
        ('cat', OneHotEncoder(), categorical_features)   # Encode categorical features
    ])

# Create complete modeling pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),                     # Preprocessing step
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  # Model
])

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print(f"Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')

# Load your dataset (using forward slashes)
df = pd.read_csv('C:/Users/narai/Downloads/Aquaculture_Project/lakshadweep_aquaculture_data_2023_enhanced.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract month from date
df['month'] = df['date'].dt.month

# Display basic info
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
df.head()

# Now create new features
print("\nCreating new features...")

# 1. Temperature-Oxygen Interaction
df['temp_oxygen_interaction'] = df['temperature'] * df['dissolved_oxygen']

# 2. Nutrient Balance
df['nutrient_balance'] = df['nitrates'] / (df['phosphates'] + 0.001)

# 3. Stress Index
df['stress_index'] = ((df['temperature'] - 28)**2 + (df['ph'] - 8.1)**2)

# 4. Seasonal Indicator
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

# 5. Total Nutrient Load
df['total_nutrients'] = df['nitrates'] + df['phosphates']

# 6. Water Quality Index (vectorized version)
oxygen_component = (df['dissolved_oxygen'] / 7) * 0.3
ph_component = (8.1 - abs(df['ph'] - 8.1)) * 0.3
ammonia_component = (1 - np.minimum(df['ammonia'] / 0.03, 1)) * 0.2
turbidity_component = (1 - np.minimum(df['turbidity'] / 10, 1)) * 0.2

df['water_quality_index'] = oxygen_component + ph_component + ammonia_component + turbidity_component

# Display the new features
print("\nNew features created:")
new_features = ['temp_oxygen_interaction', 'nutrient_balance', 'stress_index', 
               'month_sin', 'month_cos', 'total_nutrients', 'water_quality_index']
df[new_features].head()


# In[9]:


# Check if DataFrame exists, if not load it
try:
    # Check if df exists
    df.info()
except NameError:
    # Load the data
    import pandas as pd
    df = pd.read_csv('C:/Users/narai/Downloads/Aquaculture_Project/lakshadweep_aquaculture_data_2023_enhanced.csv')
    df['date'] = pd.to_datetime(df['date'])
    print("Data loaded successfully")

# Extract month from date
df['month'] = df['date'].dt.month

# Now create new features
print("Creating new features...")

# 1. Temperature-Oxygen Interaction
df['temp_oxygen_interaction'] = df['temperature'] * df['dissolved_oxygen']

# 2. Nutrient Balance
df['nutrient_balance'] = df['nitrates'] / (df['phosphates'] + 0.001)

# 3. Stress Index
df['stress_index'] = ((df['temperature'] - 28)**2 + (df['ph'] - 8.1)**2)

# 4. Seasonal Indicator
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

# 5. Total Nutrient Load
df['total_nutrients'] = df['nitrates'] + df['phosphates']

# 6. Water Quality Index (vectorized version)
oxygen_component = (df['dissolved_oxygen'] / 7) * 0.3
ph_component = (8.1 - abs(df['ph'] - 8.1)) * 0.3
ammonia_component = (1 - np.minimum(df['ammonia'] / 0.03, 1)) * 0.2
turbidity_component = (1 - np.minimum(df['turbidity'] / 10, 1)) * 0.2

df['water_quality_index'] = oxygen_component + ph_component + ammonia_component + turbidity_component

# Display the new features
print("New features created:")
new_features = ['temp_oxygen_interaction', 'nutrient_balance', 'stress_index', 
               'month_sin', 'month_cos', 'total_nutrients', 'water_quality_index']
df[new_features].head()


# In[14]:


from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

# Define models to test
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01),
    'Support Vector': SVR(kernel='rbf', C=100, gamma=0.1),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
}

# Evaluate each model with cross-validation
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Use 5-fold cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    results[name] = {
        'mean_r2': cv_scores.mean(),
        'std_r2': cv_scores.std()
    }
    print(f"{name}: Mean R² = {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['mean_r2'])
print(f"\nBest model: {best_model_name} with R² = {results[best_model_name]['mean_r2']:.3f}")


# In[15]:


# Check if engineered features exist in your DataFrame
print("Columns in DataFrame:")
print(df.columns.tolist())

# Check if engineered features are present
engineered_features = ['temp_oxygen_interaction', 'nutrient_balance', 'stress_index', 
                      'month_sin', 'month_cos', 'total_nutrients', 'water_quality_index']
missing_features = [f for f in engineered_features if f not in df.columns]
print(f"\nMissing engineered features: {missing_features}")


# In[16]:


# Define your feature set (including engineered features)
features = ['location', 'farm_type', 'temperature', 'salinity', 'dissolved_oxygen', 
            'ph', 'nitrates', 'phosphates', 'ammonia', 'chlorophyll', 'turbidity',
            'temp_oxygen_interaction', 'nutrient_balance', 'stress_index',
            'month_sin', 'month_cos', 'total_nutrients', 'water_quality_index']

# Verify features exist in DataFrame
missing = [f for f in features if f not in df.columns]
if missing:
    print(f"Missing features: {missing}")
else:
    print("All features found in DataFrame")

# Create feature matrix
X = df[features]
y = df['microbiome_health_score']


# In[17]:


# Check if target is derived from features we're using
print("Checking potential data leakage...")

# The original health score calculation used:
# disease_risk, bloom_risk, equipment_failure_risk, pH, dissolved_oxygen, ammonia
# We should avoid using these risk scores as features

# Update features to remove potential leakage
safe_features = ['location', 'farm_type', 'temperature', 'salinity', 'dissolved_oxygen', 
                 'ph', 'nitrates', 'phosphates', 'ammonia', 'chlorophyll', 'turbidity',
                 'temp_oxygen_interaction', 'nutrient_balance', 'stress_index',
                 'month_sin', 'month_cos', 'total_nutrients', 'water_quality_index']

X_safe = df[safe_features]
print(f"Using {len(safe_features)} features (removed potential leakage)")


# In[18]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Use only a few key features to start
simple_features = ['temperature', 'dissolved_oxygen', 'ph', 'ammonia']
X_simple = df[simple_features]

# Simple preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_simple)

# Train a simple model
simple_model = LinearRegression()
simple_model.fit(X_scaled, y)

# Evaluate
y_pred = simple_model.predict(X_scaled)
r2 = r2_score(y, y_pred)
print(f"Simple model R²: {r2:.3f}")

# Check coefficients
coefficients = pd.DataFrame({
    'feature': simple_features,
    'coefficient': simple_model.coef_
})
print("\nFeature coefficients:")
print(coefficients)


# In[19]:


# Check for missing values
print("Missing values per feature:")
print(df[safe_features].isnull().sum())

# Check for outliers
print("\nChecking for outliers...")
for col in ['temperature', 'dissolved_oxygen', 'ph', 'ammonia']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
    print(f"{col}: {outliers} outliers")

# Check data distributions
plt.figure(figsize=(12, 8))
for i, col in enumerate(['temperature', 'dissolved_oxygen', 'ph', 'ammonia']):
    plt.subplot(2, 2, i+1)
    df[col].hist(bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()


# In[20]:


from sklearn.model_selection import train_test_split

# Simple train-test split
X_train, X_test, y_train, y_test = train_test_split(X_safe, y, test_size=0.2, random_state=42)

# Train a model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Simple train-test split R²: {r2:.3f}")

# Compare with cross-validation
cv_scores = cross_val_score(model, X_safe, y, cv=5, scoring='r2')
print(f"Cross-validation R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")


# In[21]:


# Better neural network configuration
nn_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MLPRegressor(
        hidden_layer_sizes=(50, 25),  # Smaller layers
        max_iter=2000,                # More iterations
        early_stopping=True,          # Stop early if no improvement
        validation_fraction=0.1,      # Use 10% for validation
        n_iter_no_change=20,          # Stop if no improvement for 20 iterations
        random_state=42
    ))
])

# Train and evaluate
nn_model.fit(X_train, y_train)
y_pred = nn_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Improved Neural Network R²: {r2:.3f}")


# In[24]:


# Install if needed: !pip install xgboost
from xgboost import XGBRegressor

xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

# Train and evaluate
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"XGBoost R²: {r2:.3f}")


# In[25]:


# Get feature names after preprocessing
feature_names = numeric_features + list(model.named_steps['preprocessor']
                                      .named_transformers_['cat']
                                      .get_feature_names_out(categorical_features))

# Get feature importances
importances = model.named_steps['regressor'].feature_importances_
feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot top features
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.show()

print("Top 10 features:")
print(feature_importance.head(10))


# In[26]:


models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(50,25), max_iter=2000, early_stopping=True, random_state=42)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    results[name] = r2
    print(f"{name}: R² = {r2:.3f}")

best_model = max(results, key=results.get)
print(f"\nBest model: {best_model} with R² = {results[best_model]:.3f}")


# In[27]:


from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

# Define parameter grid for Neural Network
param_grid = {
    'regressor__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
    'regressor__activation': ['relu', 'tanh'],
    'regressor__alpha': [0.0001, 0.001, 0.01],
    'regressor__learning_rate_init': [0.001, 0.01]
}

# Create pipeline
nn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MLPRegressor(max_iter=2000, early_stopping=True, random_state=42))
])

# Grid search
grid_search = GridSearchCV(nn_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation R²: {grid_search.best_score_:.3f}")

# Evaluate on test set
best_nn = grid_search.best_estimator_
y_pred = best_nn.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
print(f"Test set R²: {test_r2:.3f}")


# In[29]:


from sklearn.ensemble import VotingRegressor

# Create individual models with optimized parameters
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
nn = MLPRegressor(hidden_layer_sizes=(50,25), activation='relu', alpha=0.0001, 
                  learning_rate_init=0.001, max_iter=2000, early_stopping=True, random_state=42)

# Create voting ensemble
voting_regressor = VotingRegressor([
    ('rf', Pipeline(steps=[('preprocessor', preprocessor), ('model', rf)])),
    ('gb', Pipeline(steps=[('preprocessor', preprocessor), ('model', gb)])),
    ('xgb', Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb)])),
    ('nn', Pipeline(steps=[('preprocessor', preprocessor), ('model', nn)]))
])

# Train and evaluate
voting_regressor.fit(X_train, y_train)
y_pred = voting_regressor.predict(X_test)
ensemble_r2 = r2_score(y_test, y_pred)
print(f"Ensemble model R²: {ensemble_r2:.3f}")


# In[31]:


# For tree-based models, we can get feature importance
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
])
rf_pipeline.fit(X_train, y_train)

# Get feature names
feature_names = numeric_features + list(rf_pipeline.named_steps['preprocessor']
                                      .named_transformers_['cat']
                                      .get_feature_names_out(categorical_features))

# Get importances
importances = rf_pipeline.named_steps['model'].feature_importances_
feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Feature Importances (Random Forest)')
plt.tight_layout()
plt.show()

# Print top features
print("Top 15 features:")
print(feature_importance.head(15))


# In[32]:


# Use the best model (Neural Network) for residual analysis
y_pred = best_nn.predict(X_test)
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(12, 6))

# Residuals vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')

# Histogram of residuals
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30, edgecolor='k')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')

plt.tight_layout()
plt.show()

# Check for patterns in residuals
print(f"Residuals statistics:")
print(f"Mean: {residuals.mean():.3f}")
print(f"Std: {residuals.std():.3f}")
print(f"Min: {residuals.min():.3f}")
print(f"Max: {residuals.max():.3f}")


# In[33]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Microbiome Health Score')
plt.tight_layout()
plt.show()


# In[34]:


# Install SHAP if needed: !pip install shap
import shap

# Use a tree-based model for SHAP (easier to interpret)
explainer_model = RandomForestRegressor(n_estimators=100, random_state=42)
explainer_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', explainer_model)
])
explainer_pipeline.fit(X_train, y_train)

# Get preprocessed data
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Create SHAP explainer
explainer = shap.TreeExplainer(explainer_model, X_train_processed)
shap_values = explainer.shap_values(X_test_processed)

# Summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, max_display=15)
plt.tight_layout()
plt.show()

# Waterfall plot for a single prediction
plt.figure(figsize=(12, 6))
shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                    base_values=explainer.expected_value,
                                    data=X_test_processed[0],
                                    feature_names=feature_names),
                   max_display=10)
plt.tight_layout()
plt.show()


# In[ ]:


import joblib

# Save the best model (Neural Network with optimized parameters)
joblib.dump(best_nn, 'best_aquaculture_model.pkl')

# Also save the preprocessor separately for future use
joblib.dump(preprocessor, 'preprocessor.pkl')

print("Model and preprocessor saved successfully!")


# In[ ]:


def predict_health_score(input_data):
    """
    Predict microbiome health score for new data.
    
    Parameters:
    input_data (DataFrame): DataFrame with the same features as training data
    
    Returns:
    array: Predicted health scores
    """
    # Load the model and preprocessor
    model = joblib.load('best_aquaculture_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    
    # Ensure input has all required features
    required_features = safe_features  # Use the safe_features list from earlier
    missing_features = set(required_features) - set(input_data.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Make prediction
    predictions = model.predict(input_data[required_features])
    return predictions

# Example usage
# Create a sample input (replace with actual data)
sample_input = X_test.head(5)
predictions = predict_health_score(sample_input)
print("Sample predictions:")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: {pred:.1f}")


# In[35]:


print("Starting hyperparameter tuning...")
# Add your hyperparameter tuning code here
print("Hyperparameter tuning completed")

print("Starting ensemble model...")
# Add your ensemble model code here
print("Ensemble model completed")

# Continue with other steps...


# In[37]:


# Add this at the beginning of your notebook
import matplotlib.pyplot as plt


# After each plot, add this to force display
plt.show()


# In[38]:


# Test basic functionality first
print("Testing basic functionality...")

# Check your data
print(f"Dataset shape: {df.shape}")
print(f"Target variable range: {df['microbiome_health_score'].min()} to {df['microbiome_health_score'].max()}")

# Test a simple model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Use only a few features for testing
test_features = ['temperature', 'dissolved_oxygen', 'ph']
X_test = df[test_features]
y_test = df['microbiome_health_score']

# Train a simple model
simple_model = RandomForestRegressor(n_estimators=10, random_state=42)
simple_model.fit(X_test, y_test)
y_pred = simple_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Simple model R²: {r2:.3f}")
print("Basic test completed successfully!")


# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set display options
pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')


print("=== Starting Analysis ===")

# Load your dataset
print("Loading dataset...")
df = pd.read_csv('C:/Users/narai/Downloads/Aquaculture_Project/lakshadweep_aquaculture_data_2023_enhanced.csv')
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Create engineered features (if not already done)
print("Creating engineered features...")
df['temp_oxygen_interaction'] = df['temperature'] * df['dissolved_oxygen']
df['nutrient_balance'] = df['nitrates'] / (df['phosphates'] + 0.001)
df['stress_index'] = ((df['temperature'] - 28)**2 + (df['ph'] - 8.1)**2)
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
df['total_nutrients'] = df['nitrates'] + df['phosphates']

# Fixed water quality index calculation
oxygen_component = (df['dissolved_oxygen'] / 7) * 0.3
ph_component = (8.1 - abs(df['ph'] - 8.1)) * 0.3
ammonia_component = (1 - np.minimum(df['ammonia'] / 0.03, 1)) * 0.2
turbidity_component = (1 - np.minimum(df['turbidity'] / 10, 1)) * 0.2

df['water_quality_index'] = oxygen_component + ph_component + ammonia_component + turbidity_component
print("Engineered features created")

# Define features and target
features = ['location', 'farm_type', 'temperature', 'salinity', 'dissolved_oxygen', 
            'ph', 'nitrates', 'phosphates', 'ammonia', 'chlorophyll', 'turbidity',
            'temp_oxygen_interaction', 'nutrient_balance', 'stress_index',
            'month_sin', 'month_cos', 'total_nutrients', 'water_quality_index']
target = 'microbiome_health_score'

X = df[features]
y = df[target]
print(f"Features selected: {len(features)} features")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

# Identify feature types
categorical_features = ['location', 'farm_type']
numeric_features = [col for col in features if col not in categorical_features]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create and train model
print("Training model...")
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.3f}")

# Feature importance
print("Analyzing feature importance...")
feature_names = numeric_features + list(model.named_steps['preprocessor']
                                      .named_transformers_['cat']
                                      .get_feature_names_out(categorical_features))
importances = model.named_steps['regressor'].feature_importances_
feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.show()

# Print top features
print("Top 10 most important features:")
print(feature_importance.head(10))

# Actual vs Predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Microbiome Health Score')
plt.tight_layout()
plt.show()

print("=== Analysis Complete ===")


# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.neural_network import MLPRegressor

# Set display options
pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')


print("=== Comprehensive Model Analysis ===")

# Load your dataset
df = pd.read_csv('C:/Users/narai/Downloads/Aquaculture_Project/lakshadweep_aquaculture_data_2023_enhanced.csv')
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Create engineered features
print("Creating engineered features...")
df['temp_oxygen_interaction'] = df['temperature'] * df['dissolved_oxygen']
df['nutrient_balance'] = df['nitrates'] / (df['phosphates'] + 0.001)
df['stress_index'] = ((df['temperature'] - 28)**2 + (df['ph'] - 8.1)**2)
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
df['total_nutrients'] = df['nitrates'] + df['phosphates']

# Fixed water quality index calculation
oxygen_component = (df['dissolved_oxygen'] / 7) * 0.3
ph_component = (8.1 - abs(df['ph'] - 8.1)) * 0.3
ammonia_component = (1 - np.minimum(df['ammonia'] / 0.03, 1)) * 0.2
turbidity_component = (1 - np.minimum(df['turbidity'] / 10, 1)) * 0.2
df['water_quality_index'] = oxygen_component + ph_component + ammonia_component + turbidity_component
print("Engineered features created")

# Define features and target
features = ['location', 'farm_type', 'temperature', 'salinity', 'dissolved_oxygen', 
            'ph', 'nitrates', 'phosphates', 'ammonia', 'chlorophyll', 'turbidity',
            'temp_oxygen_interaction', 'nutrient_balance', 'stress_index',
            'month_sin', 'month_cos', 'total_nutrients', 'water_quality_index']
target = 'microbiome_health_score'

X = df[features]
y = df[target]
print(f"Features selected: {len(features)} features")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

# Identify feature types
categorical_features = ['location', 'farm_type']
numeric_features = [col for col in features if col not in categorical_features]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])


# In[42]:


# Define models to compare
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'R²': r2}
    print(f"{name} - MSE: {mse:.2f}, R²: {r2:.3f}")

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['R²'])
print(f"\nBest model: {best_model_name} with R² = {results[best_model_name]['R²']:.3f}")


# In[43]:


# Get the best model
best_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', models[best_model_name])
])
best_model.fit(X_train, y_train)

# Get feature names after preprocessing
feature_names = numeric_features + list(best_model.named_steps['preprocessor']
                                      .named_transformers_['cat']
                                      .get_feature_names_out(categorical_features))

# Get feature importances (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    importances = best_model.named_steps['regressor'].feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title(f'Top 15 Feature Importances ({best_model_name})')
    plt.tight_layout()
    plt.show()
    
    # Print top features
    print(f"\nTop 15 most important features ({best_model_name}):")
    print(feature_importance.head(15))


# In[44]:


# Generate predictions with the best model
y_pred = best_model.predict(X_test)

# Create actual vs predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Actual vs Predicted Microbiome Health Score ({best_model_name})')
plt.tight_layout()
plt.show()

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30, edgecolor='k')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')

plt.tight_layout()
plt.show()

print(f"Residuals statistics:")
print(f"Mean: {residuals.mean():.3f}")
print(f"Std: {residuals.std():.3f}")
print(f"Min: {residuals.min():.3f}")
print(f"Max: {residuals.max():.3f}")


# In[48]:


# Create a clean results DataFrame
results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'error': y_test - y_pred,
    'abs_error': (y_test - y_pred).abs()
})

# Get original indices from test set
test_indices = X_test.index

# Add farm type using the original indices
results_df['farm_type'] = df.loc[test_indices, 'farm_type'].values

# Calculate error metrics by farm type
farm_errors = results_df.groupby('farm_type').agg({
    'abs_error': ['mean', 'std'],
    'error': ['mean', 'count']
}).round(3)

print("\nError analysis by farm type:")
print(farm_errors)

# Plot errors by farm type
plt.figure(figsize=(12, 6))
sns.boxplot(data=results_df, x='farm_type', y='abs_error')
plt.xticks(rotation=45)
plt.title('Absolute Error by Farm Type')
plt.tight_layout()
plt.show()


# In[51]:


from sklearn.model_selection import GridSearchCV

# Define parameter grid for the best model
if best_model_name == 'Random Forest':
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5]
    }
elif best_model_name == 'XGBoost':
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.01, 0.1],
        'regressor__max_depth': [3, 5]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.01, 0.1],
        'regressor__max_depth': [3, 5]
    }
else:
    param_grid = {}  # Skip optimization for neural network due to complexity

if param_grid:
    print(f"\nOptimizing {best_model_name}...")
    grid_search = GridSearchCV(
        best_model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation R²: {grid_search.best_score_:.3f}")
    
    # Evaluate optimized model
    optimized_model = grid_search.best_estimator_
    y_pred_opt = optimized_model.predict(X_test)
    r2_opt = r2_score(y_test, y_pred_opt)
    print(f"Optimized model test R²: {r2_opt:.3f}")
    
    # Update best model
    best_model = optimized_model
    best_r2 = r2_opt
else:
    best_r2 = results[best_model_name]['R²']

print(f"\nFinal best model: {best_model_name} with R² = {best_r2:.3f}")


# In[52]:


import joblib

# Save the best model
joblib.dump(best_model, 'best_aquaculture_model.pkl')
print("Best model saved successfully!")

# Save the preprocessor separately for future use
joblib.dump(preprocessor, 'preprocessor.pkl')
print("Preprocessor saved successfully!")

# Also save the feature list for future reference
import json
with open('features.json', 'w') as f:
    json.dump(features, f)
print("Feature list saved successfully!")


# In[53]:


print("=== Model Comparison ===")

# Recreate the simple model
simple_features = ['temperature', 'dissolved_oxygen', 'ph']
X_simple = df[simple_features]
y_simple = df[target]

# Split with same random state
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

# Train simple model
simple_model = RandomForestRegressor(n_estimators=100, random_state=42)
simple_model.fit(X_train_simple, y_train_simple)

# Evaluate simple model
y_pred_simple = simple_model.predict(X_test_simple)
simple_r2 = r2_score(y_test_simple, y_pred_simple)
simple_mse = mean_squared_error(y_test_simple, y_pred_simple)

print(f"Simple Model (3 features):")
print(f"R²: {simple_r2:.3f}")
print(f"MSE: {simple_mse:.2f}")

print(f"\nOptimized Model ({len(features)} features):")
print(f"R²: {best_r2:.3f}")
print(f"MSE: {mean_squared_error(y_test, best_model.predict(X_test)):.2f}")

# Check if the test sets are the same
print(f"\nTest set size - Simple: {len(y_test_simple)}, Full: {len(y_test)}")
print(f"Test sets identical: {y_test_simple.equals(y_test)}")


# In[54]:


# Get feature importance from simple model
simple_importance = pd.DataFrame({
    'feature': simple_features,
    'importance': simple_model.feature_importances_
})

# Get feature importance from optimized model
optimized_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': best_model.named_steps['regressor'].feature_importances_
}).sort_values('importance', ascending=False)

print("\nSimple Model Feature Importance:")
print(simple_importance)

print("\nOptimized Model Top 10 Features:")
print(optimized_importance.head(10))

# Plot comparison
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.barplot(x='importance', y='feature', data=simple_importance)
plt.title('Simple Model Feature Importance')
plt.xlim(0, 1)

plt.subplot(1, 2, 2)
sns.barplot(x='importance', y='feature', data=optimized_importance.head(5))
plt.title('Optimized Model Top 5 Features')
plt.xlim(0, 1)

plt.tight_layout()
plt.show()


# In[55]:


from sklearn.model_selection import cross_val_score

# Cross-validate both models
print("\n=== Cross-Validation Comparison ===")

# Simple model CV
simple_cv_scores = cross_val_score(
    simple_model, X_simple, y_simple, cv=5, scoring='r2'
)
print(f"Simple Model CV R²: {simple_cv_scores.mean():.3f} (±{simple_cv_scores.std():.3f})")

# Optimized model CV
optimized_cv_scores = cross_val_score(
    best_model, X, y, cv=5, scoring='r2'
)
print(f"Optimized Model CV R²: {optimized_cv_scores.mean():.3f} (±{optimized_cv_scores.std():.3f})")


# In[56]:


def predict_health_score(input_data):
    """
    Predict microbiome health score for new data.
    
    Parameters:
    input_data (DataFrame): DataFrame with the same features as training data
    
    Returns:
    array: Predicted health scores
    """
    # Load the model and preprocessor
    model = joblib.load('best_aquaculture_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    
    # Ensure input has all required features
    required_features = features
    missing_features = set(required_features) - set(input_data.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Make prediction
    predictions = model.predict(input_data[required_features])
    return predictions

# Test the function
print("\n=== Testing Prediction Function ===")
sample_input = X_test.head(3)
predictions = predict_health_score(sample_input)
actuals = y_test.head(3).values

for i, (pred, actual) in enumerate(zip(predictions, actuals)):
    print(f"Sample {i+1}: Predicted = {pred:.1f}, Actual = {actual:.1f}, Error = {abs(pred-actual):.1f}")


# In[57]:


print("\n=== Model Summary Report ===")
print(f"Best Model: Gradient Boosting")
print(f"Best Parameters: {best_model.named_steps['regressor'].get_params()}")
print(f"Test R²: {best_r2:.3f}")
print(f"Number of Features: {len(features)}")
print(f"Training Samples: {X_train.shape[0]}")
print(f"Test Samples: {X_test.shape[0]}")

print("\nTop 5 Most Important Features:")
for i, row in optimized_importance.head(5).iterrows():
    print(f"{i+1}. {row['feature']}: {row['importance']:.3f}")

print("\nModel Performance by Farm Type:")
for farm_type in farm_errors.index:
    mean_error = farm_errors.loc[farm_type, ('abs_error', 'mean')]
    count = farm_errors.loc[farm_type, ('error', 'count')]
    print(f"{farm_type}: Mean Error = {mean_error:.2f}, Samples = {count}")


# In[ ]:




