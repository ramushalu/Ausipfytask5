# Netflix Trend Forecasting

## 📌 Project Overview

This project focuses on analyzing historical Netflix content trends and forecasting future content additions using Machine Learning.

The project uses the Netflix dataset to identify yearly and monthly content trends, compare forecasting models, and predict the expected number of Netflix titles that may be added in future years.

---

## 🎯 Objectives

* Analyze historical Netflix content addition trends.
* Create time-based features from the `date_added` column.
* Study yearly and monthly content growth.
* Compare different forecasting models.
* Evaluate model performance using MAE, RMSE, and R².
* Forecast Netflix content additions for the next 5 years.
* Analyze future growth trends.

---

## 📂 Dataset

The project uses the Netflix dataset containing information about movies and TV shows.

### Important columns used

* `show_id`
* `type`
* `title`
* `date_added`
* `release_year`
* `rating`
* `duration`
* `country`
* `listed_in`

The `date_added` column is especially important for trend forecasting.

---

## 🔧 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Google Colab / Jupyter Notebook

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the Netflix dataset.
2. Removed duplicate records.
3. Converted `date_added` into datetime format.
4. Converted `release_year` into numeric format.
5. Removed records with missing `date_added` values.
6. Created time-based features.

### Time-Based Features

* `added_year`
* `added_month`
* `added_quarter`
* `added_day`
* `added_dayofweek`

---

## 📊 Exploratory Trend Analysis

The project analyzes Netflix content additions at different time levels.

### Yearly Analysis

The number of Netflix titles added each year was calculated to identify long-term content growth patterns.

### Monthly Analysis

Monthly content additions were analyzed to understand seasonal patterns.

### Content Type Analysis

The yearly trends of:

* Movies
* TV Shows

were also compared.

---

## 🤖 Machine Learning Models

Two forecasting models were implemented:

### 1. Linear Regression

Linear Regression was used as a baseline forecasting model to identify the general trend in Netflix content additions.

### 2. Random Forest Regression

Random Forest Regression was used to capture more complex relationships in the historical trend data.

The Random Forest model was also used for the final future forecasting.

---

## 📈 Model Evaluation

The models were evaluated using:

### MAE – Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

### RMSE – Root Mean Squared Error

Measures the square root of the average squared prediction error.

### R² Score

Measures how well the model explains the variation in the target values.

A chronological train-test split was used instead of a random split because this is a time-based forecasting problem.

---

## 🔮 Future Forecasting

After comparing the forecasting models, a final Random Forest model was trained using the available historical data.

The model forecasts Netflix content additions for the **next 5 years** after the latest year available in the dataset.

The future forecast is then analyzed to understand the expected growth pattern.

---

## 📁 Generated Files

After running the Task 5 code in Google Colab, the following files are generated:

| File                                    | Description                                               |
| --------------------------------------- | --------------------------------------------------------- |
| `netflix_forecasting_model_results.csv` | Contains model performance results                        |
| `netflix_yearly_content_trend.csv`      | Contains yearly Netflix content trends                    |
| `netflix_future_content_forecast.csv`   | Contains the predicted content additions for future years |
| `netflix_forecasting_predictions.csv`   | Contains actual and predicted values used for evaluation  |
| `netflix_trend_forecasting_model.pkl`   | Saved Random Forest forecasting model                     |

---

## 📌 Project Workflow

```text
Netflix Dataset
       ↓
Data Cleaning
       ↓
Date Processing
       ↓
Time-Based Feature Engineering
       ↓
Yearly & Monthly Trend Analysis
       ↓
Train-Test Split
       ↓
Linear Regression
       ↓
Random Forest Regression
       ↓
Model Evaluation
       ↓
Final Forecasting Model
       ↓
5-Year Future Forecast
       ↓
Growth Analysis
```

---

## 📊 Project Outputs

The project produces:

* Yearly content trend analysis
* Monthly content trend analysis
* Movie vs TV Show trend comparison
* Model performance comparison
* Actual vs predicted values
* Future Netflix content forecast
* Future growth analysis

---

## 💡 Key Learning Outcomes

Through this project, I learned:

* Time-series data preprocessing
* Feature engineering using dates
* Exploratory trend analysis
* Regression-based forecasting
* Random Forest Regression
* Model evaluation
* Future prediction
* Visualization of forecasting results
* Saving trained ML models using Joblib

---

## 🚀 Conclusion

This project demonstrates how Machine Learning can be used to analyze historical Netflix content trends and forecast future content additions.

By using time-based feature engineering and comparing Linear Regression with Random Forest Regression, the project provides a complete workflow for Netflix trend forecasting and future prediction.

---

## 👩‍💻 Project Information

**Project:** Netflix Trend Forecasting
**Task:** Task 5 – Advanced Machine Learning
**Domain:** Machine Learning / Data Science
**Platform:** Google Colab
**Language:** Python
