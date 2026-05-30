# Car Price Prediction with Machine Learning

A machine learning project that predicts the resale selling price of used cars based on features like age, mileage, fuel type, and transmission. Three regression models are trained and compared, with feature engineering to improve accuracy.

## Results

| Model | R² Score | MAE (₹ Lakhs) |
|---|---|---|
| Linear Regression | ~0.85 | ~1.2 |
| Random Forest | ~0.95 | ~0.7 |
| Gradient Boosting | ~0.96 | ~0.6 |

**Gradient Boosting is the best model**, explaining ~96% of the variance in car resale prices with an average error of under ₹0.6 Lakhs.

## Key Insights

- **Present Price** (showroom value) is the single strongest predictor of resale price
- **Car Age** has a strong negative effect — each additional year reduces resale value significantly
- **Diesel cars** retain value better than petrol cars in this dataset
- **Manual transmission** dominates the market (87% of cars in the dataset)
- Cars with **0 previous owners** command a noticeable premium

## Feature Engineering

Four new features were created beyond the raw dataset:

| Feature | Formula | Why |
|---|---|---|
| `Car_Age` | `2024 - Year` | Age is more meaningful than manufacture year |
| `Price_Depreciation` | `Present_Price - Selling_Price` | Absolute value lost |
| `Depreciation_Rate` | `Depreciation / Present_Price` | Fraction of value lost |
| `KM_Per_Year` | `Driven_kms / (Car_Age + 1)` | Usage intensity, not just total kms |

## Output Charts

| File | Description |
|---|---|
| `car_price_analysis.png` | 6-panel dashboard: distributions, correlations, age vs price, model comparison |

## Project Structure

```
car-price-prediction/
│
├── car_price_prediction.py   # Main script
├── car_data.csv              # Dataset (301 used cars)
├── car_price_analysis.png
└── README.md
```

## Dataset

- **Source:** Kaggle (Car Price Prediction dataset)
- **Rows:** 301 used cars
- **Original Features:** Car Name, Year, Selling Price, Present Price, Driven KMs, Fuel Type, Seller Type, Transmission, Owner

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Place car_data.csv in the same folder, then run:
python car_price_prediction.py
```

## Tech Stack

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (LinearRegression, RandomForestRegressor, GradientBoostingRegressor)

## Author

**Fahim Ahmed**  
2nd Year Student, Urban & Regional Planning  
Bangladesh University of Engineering and Technology (BUET)  
[LinkedIn](https://www.linkedin.com/in/fahim-ahmed-585b26357) | [GitHub](https://github.com/fmad121581-hub)
