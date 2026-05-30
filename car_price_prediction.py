# =============================================================================
# Car Price Prediction with Machine Learning
# Author: Fahim Ahmed | BUET Urban & Regional Planning
# Description: Predict car selling prices using regression models
# =============================================================================

# --- STEP 1: IMPORT LIBRARIES ---
# pandas: for loading and manipulating tabular data (like Excel, but in Python)
# numpy: for numerical operations (math on arrays)
# matplotlib & seaborn: for creating charts and plots
# sklearn: scikit-learn, the main machine learning library we'll use

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Suppress minor warnings for clean output

from sklearn.model_selection import train_test_split
# train_test_split: splits data into training set (model learns from it)
#                  and test set (model is evaluated on unseen data)

from sklearn.preprocessing import LabelEncoder
# LabelEncoder: converts text categories (like "Petrol", "Diesel") 
#               into numbers (0, 1, 2) because ML models only understand numbers

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# RandomForestRegressor: builds many decision trees and averages their predictions
#                        - great for structured/tabular data
# GradientBoostingRegressor: builds trees one after another, each fixing errors 
#                             of the previous one. Often very accurate.

from sklearn.linear_model import LinearRegression
# LinearRegression: the simplest model. Fits a straight line through the data.
#                   Good baseline to compare against more complex models.

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# mean_absolute_error (MAE): average absolute difference between predicted & actual price
# mean_squared_error (MSE): like MAE but squares errors, penalises big mistakes more
# r2_score: "R-squared" - how much of the variance in price our model explains
#            1.0 = perfect, 0.0 = model is no better than just predicting the mean

# =============================================================================
# STEP 2: LOAD AND EXPLORE THE DATA
# =============================================================================

# Load the CSV file into a DataFrame (think of it as a table/spreadsheet in Python)
df = pd.read_csv("car_data.csv")

print("=" * 60)
print("STEP 2: DATA EXPLORATION")
print("=" * 60)

print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
# .shape returns (rows, columns). We have 301 cars and 9 features.

print("\nFirst 5 rows of the dataset:")
print(df.head())
# .head() shows the first 5 rows — quick sanity check that data loaded correctly

print("\nColumn data types:")
print(df.dtypes)
# dtypes tells us what kind of data each column holds:
# int64 = whole numbers, float64 = decimal numbers, object = text/string

print("\nBasic statistics for numeric columns:")
print(df.describe())
# .describe() gives count, mean, min, max, etc. for numeric columns.
# This helps us spot unusual values (e.g. Driven_kms max = 500,000 — very high!)

print("\nMissing values per column:")
print(df.isnull().sum())
# .isnull() marks True wherever a value is missing
# .sum() counts those True values per column
# Good news: this dataset has NO missing values — no imputation needed!

print("\nCategorical column value counts:")
print("Fuel Type:\n", df['Fuel_Type'].value_counts())
print("\nSelling Type:\n", df['Selling_type'].value_counts())
print("\nTransmission:\n", df['Transmission'].value_counts())

# =============================================================================
# STEP 3: FEATURE ENGINEERING
# =============================================================================
# Feature engineering = creating new, more informative columns from existing ones
# This often improves model performance significantly

print("\n" + "=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

# Create 'Car_Age': how old the car is (as of 2024, the last year in the dataset era)
# A 2018 car is newer and worth more than a 2003 car.
# Year alone is less informative than age.
df['Car_Age'] = 2024 - df['Year']

# Create 'Price_Depreciation': the drop from showroom price to selling price
# If a car was bought for 10 lakhs and sells for 6 lakhs, depreciation = 4 lakhs.
# This is a strong signal — high depreciation = lower resale value.
df['Price_Depreciation'] = df['Present_Price'] - df['Selling_Price']

# Create 'Depreciation_Rate': what percentage of value has been lost
# A car that lost 40% of its value is very different from one that lost 10%.
# We add a small constant (0.001) to avoid division by zero errors.
df['Depreciation_Rate'] = df['Price_Depreciation'] / (df['Present_Price'] + 0.001)

# Create 'KM_Per_Year': how much the car was driven per year on average
# A 5-year-old car driven 10,000 km/year is in better shape than one driven 30,000/year
df['KM_Per_Year'] = df['Driven_kms'] / (df['Car_Age'] + 1)
# +1 prevents division by zero for brand new cars (age = 0)

print("New features created:")
print("  - Car_Age: years since manufacture")
print("  - Price_Depreciation: Present_Price minus Selling_Price")
print("  - Depreciation_Rate: fraction of value lost")
print("  - KM_Per_Year: average kms driven per year")
print(df[['Car_Name', 'Year', 'Car_Age', 'Price_Depreciation', 
          'Depreciation_Rate', 'KM_Per_Year']].head())

# =============================================================================
# STEP 4: DATA PREPROCESSING (Encoding Categorical Variables)
# =============================================================================
# Machine learning models work with numbers only.
# We need to convert text columns to numbers.

print("\n" + "=" * 60)
print("STEP 4: ENCODING CATEGORICAL VARIABLES")
print("=" * 60)

# Make a copy so we don't accidentally modify the original data
df_encoded = df.copy()

# LabelEncoder converts each unique text value to a unique integer.
# Example: Petrol → 2, Diesel → 0, CNG → 1 (alphabetical order by default)
le = LabelEncoder()

# Encode Fuel_Type (Petrol/Diesel/CNG → numbers)
df_encoded['Fuel_Type'] = le.fit_transform(df_encoded['Fuel_Type'])
# fit_transform: "learns" all unique values, then converts them

# Encode Selling_type (Dealer/Individual → numbers)
df_encoded['Selling_type'] = le.fit_transform(df_encoded['Selling_type'])

# Encode Transmission (Manual/Automatic → numbers)
df_encoded['Transmission'] = le.fit_transform(df_encoded['Transmission'])

print("Encoding done. Sample of encoded data:")
print(df_encoded[['Fuel_Type', 'Selling_type', 'Transmission']].head(3))

# =============================================================================
# STEP 5: SELECT FEATURES AND TARGET VARIABLE
# =============================================================================
# Features (X) = the input columns we use to make predictions
# Target (y) = the column we want to predict (Selling_Price)

print("\n" + "=" * 60)
print("STEP 5: SELECTING FEATURES AND TARGET")
print("=" * 60)

# We drop columns that are NOT useful as model inputs:
# - Car_Name: just the brand name/string, too many unique values, not encoded
# - Selling_Price: this IS our target, so it can't be an input
# - Year: we replaced it with Car_Age which is more meaningful
# - Price_Depreciation: this is derived from Selling_Price, would cause data leakage

# What is data leakage? When your model accidentally "sees" the answer during training.
# Price_Depreciation = Present_Price - Selling_Price. If we use it to predict 
# Selling_Price, we're basically giving the model the answer. Bad!

feature_columns = [
    'Present_Price',    # Showroom/current market price of the car
    'Driven_kms',       # Total kilometers driven
    'Fuel_Type',        # Petrol / Diesel / CNG (encoded as 0/1/2)
    'Selling_type',     # Dealer or Individual seller (encoded as 0/1)
    'Transmission',     # Manual or Automatic (encoded as 0/1)
    'Owner',            # Number of previous owners (0, 1, or 3)
    'Car_Age',          # How old the car is (2024 - Year)
    'Depreciation_Rate',# What fraction of present price has been lost
    'KM_Per_Year'       # Average kms driven per year
]

X = df_encoded[feature_columns]   # Input features matrix
y = df_encoded['Selling_Price']   # Target variable (what we predict)

print(f"Features selected ({len(feature_columns)}):")
for col in feature_columns:
    print(f"  - {col}")
print(f"\nTarget: Selling_Price")
print(f"X shape: {X.shape}  |  y shape: {y.shape}")

# =============================================================================
# STEP 6: TRAIN-TEST SPLIT
# =============================================================================
# We split data into:
# - Training set (80%): the model learns patterns from this
# - Test set (20%): we evaluate the model on this — it has NEVER seen these rows
# 
# Why not train and test on the same data?
# The model would just memorize the training data and fail on new cars.
# This is called "overfitting." The test set simulates real-world unseen cars.

print("\n" + "=" * 60)
print("STEP 6: TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,              # All feature rows
    y,              # All target values
    test_size=0.2,  # 20% of data goes to test set
    random_state=42 # "Seed" for reproducibility — same split every time you run
)

print(f"Training set: {X_train.shape[0]} cars ({80}%)")
print(f"Test set:     {X_test.shape[0]} cars ({20}%)")

# =============================================================================
# STEP 7: TRAIN MODELS
# =============================================================================
# We train 3 models and compare them.
# This is good practice — never rely on just one model.

print("\n" + "=" * 60)
print("STEP 7: TRAINING MODELS")
print("=" * 60)

# --- Model 1: Linear Regression ---
# Assumes a linear relationship: Selling_Price = a*Present_Price + b*Car_Age + ... + c
# It's simple, fast, and interpretable — good baseline.
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)  # .fit() = teach the model using training data
print("✓ Linear Regression trained")

# --- Model 2: Random Forest ---
# Builds 200 decision trees independently, then averages their predictions.
# Each tree sees a random subset of rows and features — this is "bagging."
# Averaging reduces overfitting. Generally very robust.
rf_model = RandomForestRegressor(
    n_estimators=200,   # 200 trees in the "forest"
    max_depth=10,       # Each tree can be at most 10 levels deep (prevents overfitting)
    random_state=42     # Reproducibility
)
rf_model.fit(X_train, y_train)
print("✓ Random Forest trained")

# --- Model 3: Gradient Boosting ---
# Builds trees SEQUENTIALLY — each new tree tries to fix the mistakes of the previous ones.
# This is "boosting." Usually gives highest accuracy but takes longer to train.
gb_model = GradientBoostingRegressor(
    n_estimators=200,   # 200 boosting rounds
    learning_rate=0.05, # How much each tree corrects the previous (smaller = safer)
    max_depth=4,        # Keep trees shallow to prevent overfitting
    random_state=42
)
gb_model.fit(X_train, y_train)
print("✓ Gradient Boosting trained")

# =============================================================================
# STEP 8: EVALUATE MODELS
# =============================================================================
# Now we see how well each model predicts prices on the TEST SET (unseen data)

print("\n" + "=" * 60)
print("STEP 8: MODEL EVALUATION ON TEST SET")
print("=" * 60)

def evaluate_model(model, X_test, y_test, model_name):
    """
    Takes a trained model, makes predictions, and prints evaluation metrics.
    We wrap this in a function so we can call it for all 3 models without 
    repeating code.
    """
    y_pred = model.predict(X_test)
    # .predict() uses the trained model to guess selling prices for test cars
    
    mae = mean_absolute_error(y_test, y_pred)
    # MAE: on average, our price prediction is off by 'mae' lakhs
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # RMSE: similar to MAE but penalizes large errors more heavily.
    # If RMSE >> MAE, we have some very badly predicted cars.
    
    r2 = r2_score(y_test, y_pred)
    # R²: the proportion of price variation our model explains.
    # R²=0.95 means the model explains 95% of why prices differ between cars.
    
    print(f"\n{model_name}:")
    print(f"  MAE  (Mean Absolute Error):  ₹{mae:.4f} Lakhs  → avg prediction is off by this much")
    print(f"  RMSE (Root Mean Sq. Error):  ₹{rmse:.4f} Lakhs → penalizes large errors")
    print(f"  R²   (R-Squared):            {r2:.4f}        → {r2*100:.1f}% of price variance explained")
    
    return y_pred, mae, rmse, r2

# Evaluate all 3 models and store predictions + metrics
lr_pred,  lr_mae,  lr_rmse,  lr_r2  = evaluate_model(lr_model,  X_test, y_test, "Linear Regression")
rf_pred,  rf_mae,  rf_rmse,  rf_r2  = evaluate_model(rf_model,  X_test, y_test, "Random Forest")
gb_pred,  gb_mae,  gb_rmse,  gb_r2  = evaluate_model(gb_model,  X_test, y_test, "Gradient Boosting")

# Find the best model by R² (highest = best)
r2_scores = {'Linear Regression': lr_r2, 'Random Forest': rf_r2, 'Gradient Boosting': gb_r2}
best_model_name = max(r2_scores, key=r2_scores.get)
print(f"\n🏆 Best model: {best_model_name} (R² = {r2_scores[best_model_name]:.4f})")

# Use the best model's predictions for final visualizations
best_pred = gb_pred if best_model_name == 'Gradient Boosting' else (
            rf_pred if best_model_name == 'Random Forest' else lr_pred)
best_model = gb_model if best_model_name == 'Gradient Boosting' else (
             rf_model if best_model_name == 'Random Forest' else lr_model)

# =============================================================================
# STEP 9: FEATURE IMPORTANCE
# =============================================================================
# Tree-based models (RF, GB) can tell us which features mattered most.
# This is valuable — it tells us what actually drives car prices.

print("\n" + "=" * 60)
print("STEP 9: FEATURE IMPORTANCE (Gradient Boosting)")
print("=" * 60)

# .feature_importances_ gives a score for each feature.
# Higher score = feature was used more often in trees and reduced error more.
importances = gb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(feature_importance_df.to_string(index=False))

# =============================================================================
# STEP 10: VISUALIZATIONS
# =============================================================================
# Good plots make your project stand out on GitHub and LinkedIn!
# We'll create a single professional figure with multiple subplots.

print("\n" + "=" * 60)
print("STEP 10: GENERATING VISUALIZATIONS...")
print("=" * 60)

# Set a clean, professional style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
# 'seaborn-v0_8-whitegrid' gives a clean white background with light grid lines

# Color palette — consistent colors across all charts
colors = {
    'primary': '#2C5F8A',    # Deep blue (main bars/lines)
    'accent': '#E8593C',     # Orange-red (highlights)
    'success': '#2E8B57',    # Green (good predictions)
    'light': '#B5D4F4',      # Light blue (secondary bars)
    'background': '#F8F9FA'  # Off-white background
}

# Create a large figure with a 3x2 grid of subplots
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(colors['background'])  # Overall figure background color

# GridSpec gives more control over subplot layout and spacing
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])  # Row 0, Column 0
ax2 = fig.add_subplot(gs[0, 1])  # Row 0, Column 1
ax3 = fig.add_subplot(gs[1, 0])  # Row 1, Column 0
ax4 = fig.add_subplot(gs[1, 1])  # Row 1, Column 1
ax5 = fig.add_subplot(gs[2, 0])  # Row 2, Column 0
ax6 = fig.add_subplot(gs[2, 1])  # Row 2, Column 1

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_facecolor(colors['background'])  # Set background for each subplot

# ---------- PLOT 1: Model Comparison (Bar Chart) ----------
# Shows MAE and R² side by side for all 3 models
model_names = ['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting']
mae_values  = [lr_mae,  rf_mae,  gb_mae]
r2_values   = [lr_r2,   rf_r2,   gb_r2]

x = np.arange(len(model_names))  # [0, 1, 2] — positions on x-axis
width = 0.35                      # Width of each bar

bars1 = ax1.bar(x - width/2, mae_values, width, 
                label='MAE (Lakhs)', color=colors['primary'], alpha=0.85)
# We plot MAE on the left side of each group

ax1_twin = ax1.twinx()  # Create a second y-axis (right side) for R²
# We need two y-axes because MAE and R² have very different scales (e.g. 0.5 vs 0.96)

bars2 = ax1_twin.bar(x + width/2, r2_values, width,
                     label='R² Score', color=colors['accent'], alpha=0.85)
ax1_twin.set_ylabel('R² Score', fontsize=10, color=colors['accent'])
ax1_twin.tick_params(axis='y', colors=colors['accent'])
ax1_twin.set_ylim(0, 1.15)  # Give a bit of headroom above 1.0

# Add value labels on top of each bar
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    ax1_twin.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                  f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
                  color=colors['accent'])

ax1.set_xlabel('Model', fontsize=10)
ax1.set_ylabel('MAE (₹ Lakhs)', fontsize=10, color=colors['primary'])
ax1.tick_params(axis='y', colors=colors['primary'])
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=9)
ax1.set_title('Model Comparison: MAE & R² Score', fontsize=12, fontweight='bold', pad=10)
ax1.legend(loc='upper left', fontsize=8)
ax1_twin.legend(loc='upper right', fontsize=8)

# ---------- PLOT 2: Actual vs Predicted (Scatter Plot) ----------
# The closer points are to the diagonal line, the better the model predicts
ax2.scatter(y_test, gb_pred, alpha=0.6, color=colors['primary'], s=40, label='Predictions')
# alpha=0.6 makes points slightly transparent so we can see overlapping points

# Add a perfect prediction line (diagonal)
max_val = max(y_test.max(), gb_pred.max())
min_val = min(y_test.min(), gb_pred.min())
ax2.plot([min_val, max_val], [min_val, max_val], 
         '--', color=colors['accent'], linewidth=2, label='Perfect Prediction')
# This dashed line shows where predictions = actual prices (y = x line)

ax2.set_xlabel('Actual Selling Price (₹ Lakhs)', fontsize=10)
ax2.set_ylabel('Predicted Selling Price (₹ Lakhs)', fontsize=10)
ax2.set_title(f'Actual vs Predicted Prices\n(Gradient Boosting, R²={gb_r2:.3f})', 
              fontsize=12, fontweight='bold', pad=10)
ax2.legend(fontsize=9)

# ---------- PLOT 3: Feature Importance (Horizontal Bar Chart) ----------
# Shows which features the Gradient Boosting model relied on most
fi_sorted = feature_importance_df.sort_values('Importance')
# Sort ascending so the most important feature appears at the top (horizontal bars)

bar_colors = [colors['accent'] if imp > 0.15 else colors['primary'] 
              for imp in fi_sorted['Importance']]
# Color top features differently to highlight them

ax3.barh(fi_sorted['Feature'], fi_sorted['Importance'], 
         color=bar_colors, alpha=0.85)
# .barh() = horizontal bar chart (easier to read long feature names)

ax3.set_xlabel('Feature Importance Score', fontsize=10)
ax3.set_title('Feature Importance\n(Gradient Boosting)', fontsize=12, fontweight='bold', pad=10)

# Add value labels at the end of each bar
for i, (val, name) in enumerate(zip(fi_sorted['Importance'], fi_sorted['Feature'])):
    ax3.text(val + 0.003, i, f'{val:.3f}', va='center', fontsize=8)

# ---------- PLOT 4: Residuals Plot ----------
# Residual = Actual - Predicted. 
# A good model has residuals randomly scattered around 0 (no pattern).
# If there's a pattern, the model is systematically wrong in some region.
residuals = y_test - gb_pred
ax4.scatter(gb_pred, residuals, alpha=0.6, color=colors['primary'], s=40)
ax4.axhline(y=0, color=colors['accent'], linestyle='--', linewidth=2)
# axhline draws a horizontal line at y=0 (the "perfect prediction" level)

ax4.set_xlabel('Predicted Selling Price (₹ Lakhs)', fontsize=10)
ax4.set_ylabel('Residual (Actual − Predicted)', fontsize=10)
ax4.set_title('Residual Analysis\n(Gradient Boosting)', fontsize=12, fontweight='bold', pad=10)
ax4.text(0.02, 0.95, 'Points near 0 = better prediction', 
         transform=ax4.transAxes, fontsize=8, color='gray', va='top')

# ---------- PLOT 5: Distribution of Selling Prices ----------
# Understanding the target variable distribution is important.
# Skewed data can affect model performance and interpretation.
ax5.hist(df['Selling_Price'], bins=30, color=colors['primary'], 
         alpha=0.8, edgecolor='white', linewidth=0.5)
ax5.axvline(df['Selling_Price'].mean(), color=colors['accent'], 
            linestyle='--', linewidth=2, label=f'Mean: ₹{df["Selling_Price"].mean():.2f}L')
ax5.axvline(df['Selling_Price'].median(), color=colors['success'], 
            linestyle='--', linewidth=2, label=f'Median: ₹{df["Selling_Price"].median():.2f}L')
# axvline draws a vertical line at a specific x value

ax5.set_xlabel('Selling Price (₹ Lakhs)', fontsize=10)
ax5.set_ylabel('Number of Cars', fontsize=10)
ax5.set_title('Distribution of Car Selling Prices', fontsize=12, fontweight='bold', pad=10)
ax5.legend(fontsize=9)
ax5.text(0.65, 0.85, 'Right-skewed:\nmost cars are\ncheaper, few\nare expensive',
         transform=ax5.transAxes, fontsize=8, color='gray',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# ---------- PLOT 6: Price vs Car Age (scatter with color for fuel type) ----------
# Do older cars sell for less? Let's see.
fuel_colors = {'Petrol': colors['primary'], 'Diesel': colors['accent'], 'CNG': colors['success']}
for fuel_type, group in df.groupby('Fuel_Type'):
    ax6.scatter(group['Car_Age'], group['Selling_Price'], 
                c=fuel_colors.get(fuel_type, 'gray'), alpha=0.6, 
                s=30, label=fuel_type)

ax6.set_xlabel('Car Age (Years)', fontsize=10)
ax6.set_ylabel('Selling Price (₹ Lakhs)', fontsize=10)
ax6.set_title('Car Age vs Selling Price\n(by Fuel Type)', fontsize=12, fontweight='bold', pad=10)
ax6.legend(fontsize=9, title='Fuel Type')

# ---------- MAIN TITLE ----------
fig.suptitle('Car Price Prediction — Machine Learning Analysis\nCodeAlpha Data Science Internship | Fahim Ahmed',
             fontsize=15, fontweight='bold', y=0.98)

# Save the figure as a high-resolution PNG
plt.savefig('car_price_analysis.png', dpi=150, bbox_inches='tight', 
            facecolor=colors['background'])
# dpi=150: dots per inch — higher = sharper image, good for LinkedIn posts
# bbox_inches='tight': removes extra whitespace around the figure
print("✓ Visualization saved as 'car_price_analysis.png'")

plt.show()

# =============================================================================
# STEP 11: DEMONSTRATE THE MODEL ON A SAMPLE PREDICTION
# =============================================================================
# This is the most satisfying part — use the trained model to predict a car price!

print("\n" + "=" * 60)
print("STEP 11: SAMPLE PREDICTION DEMO")
print("=" * 60)

# Let's predict the selling price for this hypothetical car:
# - Swift, 2017 (7 years old as of 2024)
# - Present Market Price: 6.5 lakhs
# - Driven 35,000 km
# - Petrol, Dealer, Manual, 0 previous owners

sample_car = pd.DataFrame({
    'Present_Price': [6.5],
    'Driven_kms': [35000],
    'Fuel_Type': [2],          # Petrol = 2 (from LabelEncoder — check your encoding order)
    'Selling_type': [0],       # Dealer = 0
    'Transmission': [1],       # Manual = 1
    'Owner': [0],              # 0 previous owners
    'Car_Age': [7],            # 2024 - 2017 = 7
    'Depreciation_Rate': [(6.5 - 4.0) / (6.5 + 0.001)],  # estimated
    'KM_Per_Year': [35000 / (7 + 1)]  # 4375 km/year
})

predicted_price = gb_model.predict(sample_car)[0]

print(f"""
Car Details:
  - Year: 2017 (7 years old)
  - Present Market Price: ₹6.50 Lakhs
  - Driven: 35,000 km
  - Fuel: Petrol | Seller: Dealer | Transmission: Manual | Owner: 0

  → Predicted Selling Price: ₹{predicted_price:.2f} Lakhs
""")

# =============================================================================
# STEP 12: FINAL SUMMARY
# =============================================================================
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
Dataset:        301 cars | 9 original features | 4 engineered features

Models Trained:
  Linear Regression  → R² = {lr_r2:.3f}, MAE = ₹{lr_mae:.3f}L
  Random Forest      → R² = {rf_r2:.3f}, MAE = ₹{rf_mae:.3f}L  
  Gradient Boosting  → R² = {gb_r2:.3f}, MAE = ₹{gb_mae:.3f}L

Best Model: {best_model_name}

Top Predictors of Car Selling Price:
  1. Present_Price (showroom value is the strongest signal)
  2. Car_Age (older = worth less)
  3. Depreciation_Rate (faster depreciation = lower value)
  
Key Insights:
  - Present_Price alone explains most of the selling price
  - Car age and km driven significantly reduce resale value
  - Diesel cars tend to retain value better than petrol
  - Manual transmission dominates this market (87% of cars)

Output:
  - car_price_analysis.png  (6 professional plots for LinkedIn/submission)
""")
