# TASK 5 - NETFLIX TREND FORECASTING MODEL

# Machine Learning Internship Project
# Objective:
# Forecast future Netflix content release trends based on
# historical Netflix content data.
# Workflow:
# 1. Prepare time-based features
# 2. Analyze historical release patterns
# 3. Build forecasting models
# 4. Generate future predictions
# 5. Evaluate forecasting accuracy
# Models:
# - Linear Regression
# - Random Forest Regression
# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from IPython.display import display

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


print("=" * 70)
print("TASK 5 - NETFLIX TREND FORECASTING MODEL")
print("=" * 70)

# 2. LOAD DATASET

df = pd.read_csv("Dataset.csv")

print("\nDataset loaded successfully!")

print("Dataset shape:", df.shape)

# 3. BASIC DATA EXPLORATION

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nFirst 5 rows:")

display(df.head())


print("\nColumn names:")

print(df.columns.tolist())


print("\nMissing values:")

print(df.isnull().sum())


print("\nDuplicate rows:")

print(df.duplicated().sum())

# 4. DATA CLEANING

df = df.drop_duplicates().copy()


# Convert date_added into datetime

df["date_added"] = pd.to_datetime(
    df["date_added"],
    errors="coerce"
)


# Convert release_year into numeric

df["release_year"] = pd.to_numeric(
    df["release_year"],
    errors="coerce"
)


# Remove rows where date_added is missing

df = df.dropna(
    subset=["date_added"]
).copy()


print("\nAfter cleaning:")

print("Dataset shape:", df.shape)

# 5. CREATE TIME-BASED FEATURES

df["added_year"] = (
    df["date_added"].dt.year
)

df["added_month"] = (
    df["date_added"].dt.month
)

df["added_quarter"] = (
    df["date_added"].dt.quarter
)

df["added_day"] = (
    df["date_added"].dt.day
)

df["added_dayofweek"] = (
    df["date_added"].dt.dayofweek
)


print("\n" + "=" * 70)
print("TIME-BASED FEATURES CREATED")
print("=" * 70)

print("- added_year")
print("- added_month")
print("- added_quarter")
print("- added_day")
print("- added_dayofweek")

# 6. HISTORICAL YEARLY CONTENT ANALYSIS

yearly_content = (
    df.groupby("added_year")
      .size()
      .reset_index(name="content_count")
)


yearly_content = yearly_content.sort_values(
    "added_year"
)


print("\n" + "=" * 70)
print("YEARLY NETFLIX CONTENT ADDITIONS")
print("=" * 70)

display(yearly_content)

# 7. YEARLY TREND VISUALIZATION

plt.figure(figsize=(12, 6))

plt.plot(
    yearly_content["added_year"],
    yearly_content["content_count"],
    marker="o"
)

plt.title(
    "Netflix Content Added Per Year"
)

plt.xlabel("Year")

plt.ylabel("Number of Titles Added")

plt.grid(True)

plt.tight_layout()

plt.show()

# 8. MONTHLY CONTENT ANALYSIS

monthly_content = (
    df.groupby("added_month")
      .size()
)


print("\n" + "=" * 70)
print("MONTHLY NETFLIX CONTENT ADDITIONS")
print("=" * 70)

print(monthly_content)

# 9. MONTHLY TREND GRAPH

plt.figure(figsize=(12, 6))

plt.bar(
    monthly_content.index,
    monthly_content.values
)

plt.title(
    "Netflix Content Added by Month"
)

plt.xlabel("Month")

plt.ylabel("Number of Titles")

plt.xticks(
    range(1, 13)
)

plt.tight_layout()

plt.show()

# 10. MOVIE VS TV SHOW YEARLY TREND

type_yearly = (
    df.groupby(
        ["added_year", "type"]
    )
    .size()
    .unstack(fill_value=0)
)


print("\n" + "=" * 70)
print("MOVIE VS TV SHOW YEARLY TREND")
print("=" * 70)

display(type_yearly)

# 11. MOVIE VS TV SHOW TREND GRAPH

plt.figure(figsize=(12, 6))

for content_type in type_yearly.columns:

    plt.plot(
        type_yearly.index,
        type_yearly[content_type],
        marker="o",
        label=content_type
    )


plt.title(
    "Netflix Movies vs TV Shows Added Per Year"
)

plt.xlabel("Year")

plt.ylabel("Number of Titles")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# 12. PREPARE FORECASTING DATA

forecast_data = yearly_content.copy()


# Create sequential time index

forecast_data["time_index"] = np.arange(
    len(forecast_data)
)


print("\n" + "=" * 70)
print("FORECASTING DATA")
print("=" * 70)

display(forecast_data)

# 13. DEFINE FEATURES AND TARGET

X = forecast_data[
    [
        "time_index"
    ]
]

y = forecast_data[
    "content_count"
]

# 14. TRAIN-TEST SPLIT

# Time series data should be split chronologically.
# The earlier observations are used for training and
# later observations are used for testing.

split_index = int(
    len(forecast_data) * 0.80
)


X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]


y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


print("\n" + "=" * 70)
print("TIME-BASED TRAIN-TEST SPLIT")
print("=" * 70)

print(
    "Training years:",
    forecast_data["added_year"].iloc[0],
    "to",
    forecast_data["added_year"].iloc[
        split_index - 1
    ]
)

print(
    "Testing years:",
    forecast_data["added_year"].iloc[
        split_index
    ],
    "to",
    forecast_data["added_year"].iloc[-1]
)

# 15. LINEAR REGRESSION FORECASTING MODEL

print("\n" + "=" * 70)
print("TRAINING LINEAR REGRESSION")
print("=" * 70)


linear_model = LinearRegression()


linear_model.fit(
    X_train,
    y_train
)


linear_predictions = linear_model.predict(
    X_test
)

# 16. LINEAR REGRESSION EVALUATION

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)


linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)


linear_r2 = r2_score(
    y_test,
    linear_predictions
)


print(
    "Linear Regression MAE:",
    round(linear_mae, 2)
)

print(
    "Linear Regression RMSE:",
    round(linear_rmse, 2)
)

print(
    "Linear Regression R2:",
    round(linear_r2, 4)
)

# 17. RANDOM FOREST REGRESSION MODEL

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST REGRESSION")
print("=" * 70)


random_forest_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)


random_forest_model.fit(
    X_train,
    y_train
)


random_forest_predictions = (
    random_forest_model.predict(
        X_test
    )
)

# 18. RANDOM FOREST EVALUATION

rf_mae = mean_absolute_error(
    y_test,
    random_forest_predictions
)


rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        random_forest_predictions
    )
)


rf_r2 = r2_score(
    y_test,
    random_forest_predictions
)


print(
    "Random Forest MAE:",
    round(rf_mae, 2)
)

print(
    "Random Forest RMSE:",
    round(rf_rmse, 2)
)

print(
    "Random Forest R2:",
    round(rf_r2, 4)
)

# 19. MODEL COMPARISON

print("\n" + "=" * 70)
print("FORECASTING MODEL COMPARISON")
print("=" * 70)


model_results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest Regression"
    ],

    "MAE": [
        linear_mae,
        rf_mae
    ],

    "RMSE": [
        linear_rmse,
        rf_rmse
    ],

    "R2_Score": [
        linear_r2,
        rf_r2
    ]

})


display(
    model_results.round(4)
)

# 20. ACTUAL VS PREDICTED VALUES

prediction_comparison = pd.DataFrame({

    "Year": forecast_data[
        "added_year"
    ].iloc[
        split_index:
    ].values,

    "Actual_Content_Count": y_test.values,

    "Linear_Regression_Prediction":
        np.round(
            linear_predictions
        ),

    "Random_Forest_Prediction":
        np.round(
            random_forest_predictions
        )

})


print("\n" + "=" * 70)
print("ACTUAL VS PREDICTED CONTENT")
print("=" * 70)

display(
    prediction_comparison
)

# 21. ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(12, 6))


plt.plot(
    prediction_comparison["Year"],
    prediction_comparison[
        "Actual_Content_Count"
    ],
    marker="o",
    label="Actual"
)


plt.plot(
    prediction_comparison["Year"],
    prediction_comparison[
        "Linear_Regression_Prediction"
    ],
    marker="o",
    label="Linear Regression"
)


plt.plot(
    prediction_comparison["Year"],
    prediction_comparison[
        "Random_Forest_Prediction"
    ],
    marker="o",
    label="Random Forest"
)


plt.title(
    "Actual vs Predicted Netflix Content Additions"
)

plt.xlabel("Year")

plt.ylabel("Number of Titles")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# 22. TRAIN FINAL MODEL USING ALL HISTORICAL DATA

print("\n" + "=" * 70)
print("TRAINING FINAL FORECASTING MODEL")
print("=" * 70)


# Use Random Forest for the final forecasting model

final_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)


final_model.fit(
    X,
    y
)


print(
    "Final forecasting model trained!"
)

# 23. GENERATE FUTURE YEARS

last_year = int(
    forecast_data["added_year"].max()
)


future_years = np.arange(
    last_year + 1,
    last_year + 6
)


future_time_index = np.arange(
    len(forecast_data),
    len(forecast_data) + 5
)


future_X = pd.DataFrame({

    "time_index":
        future_time_index

})


future_predictions = final_model.predict(
    future_X
)


future_predictions = np.maximum(
    0,
    np.round(
        future_predictions
    )
)

# 24. FUTURE FORECAST RESULTS

future_forecast = pd.DataFrame({

    "Year":
        future_years,

    "Predicted_Content_Additions":
        future_predictions.astype(int)

})


print("\n" + "=" * 70)
print("FUTURE NETFLIX CONTENT FORECAST")
print("=" * 70)

display(
    future_forecast
)

# 25. HISTORICAL + FUTURE FORECAST GRAPH

plt.figure(figsize=(13, 7))


plt.plot(
    forecast_data["added_year"],
    forecast_data["content_count"],
    marker="o",
    label="Historical Content"
)


plt.plot(
    future_forecast["Year"],
    future_forecast[
        "Predicted_Content_Additions"
    ],
    marker="o",
    linestyle="--",
    label="Future Forecast"
)


plt.axvline(
    last_year,
    linestyle=":"
)


plt.title(
    "Netflix Content Trend Forecast"
)

plt.xlabel("Year")

plt.ylabel("Number of Content Additions")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# 26. FORECAST GROWTH ANALYSIS

first_forecast = future_forecast[
    "Predicted_Content_Additions"
].iloc[0]


last_forecast = future_forecast[
    "Predicted_Content_Additions"
].iloc[-1]


if first_forecast > 0:

    forecast_growth = (
        (
            last_forecast -
            first_forecast
        )
        / first_forecast
    ) * 100

else:

    forecast_growth = 0


print("\n" + "=" * 70)
print("FORECAST TREND ANALYSIS")
print("=" * 70)


print(
    "First forecast year:",
    future_years[0]
)

print(
    "Last forecast year:",
    future_years[-1]
)

print(
    "Predicted content in first forecast year:",
    int(first_forecast)
)

print(
    "Predicted content in last forecast year:",
    int(last_forecast)
)

print(
    "Forecast change:",
    round(forecast_growth, 2),
    "%"
)

# 27. MONTHLY HISTORICAL TREND

monthly_year = (
    df.groupby(
        [
            "added_year",
            "added_month"
        ]
    )
    .size()
    .reset_index(
        name="content_count"
    )
)


print("\n" + "=" * 70)
print("HISTORICAL MONTHLY CONTENT DATA")
print("=" * 70)

display(
    monthly_year.head(20)
)

# 28. CONTENT TYPE TREND

movie_count = (
    df[df["type"] == "Movie"]
    .groupby("added_year")
    .size()
)


tv_count = (
    df[df["type"] == "TV Show"]
    .groupby("added_year")
    .size()
)


print("\n" + "=" * 70)
print("CONTENT TYPE TREND")
print("=" * 70)


print(
    "Total Movies:",
    int(
        (df["type"] == "Movie").sum()
    )
)


print(
    "Total TV Shows:",
    int(
        (df["type"] == "TV Show").sum()
    )
)

# 29. CONTENT TYPE TREND GRAPH

plt.figure(figsize=(12, 6))


plt.plot(
    movie_count.index,
    movie_count.values,
    marker="o",
    label="Movies"
)


plt.plot(
    tv_count.index,
    tv_count.values,
    marker="o",
    label="TV Shows"
)


plt.title(
    "Netflix Movie and TV Show Release Trend"
)

plt.xlabel("Year")

plt.ylabel("Number of Titles")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# 30. SAVE MODEL RESULTS

model_results.to_csv(
    "netflix_forecasting_model_results.csv",
    index=False
)

# 31. SAVE HISTORICAL TREND DATA

yearly_content.to_csv(
    "netflix_yearly_content_trend.csv",
    index=False
)

# 32. SAVE FUTURE FORECAST

future_forecast.to_csv(
    "netflix_future_content_forecast.csv",
    index=False
)

# 33. SAVE PREDICTION COMPARISON

prediction_comparison.to_csv(
    "netflix_forecasting_predictions.csv",
    index=False
)

# 34. SAVE FINAL MODEL

joblib.dump(
    final_model,
    "netflix_trend_forecasting_model.pkl"
)


print("\n" + "=" * 70)
print("FILES SAVED SUCCESSFULLY")
print("=" * 70)

print(
    "1. netflix_forecasting_model_results.csv"
)

print(
    "2. netflix_yearly_content_trend.csv"
)

print(
    "3. netflix_future_content_forecast.csv"
)

print(
    "4. netflix_forecasting_predictions.csv"
)

print(
    "5. netflix_trend_forecasting_model.pkl"
)

# 35. FINAL PROJECT SUMMARY

print("\n")
print("=" * 70)
print("TASK 5 - PROJECT SUMMARY")
print("=" * 70)


print(
    "Project: Netflix Trend Forecasting Model"
)


print(
    "\nObjective:"
)


print(
    "Forecast future Netflix content release trends "
    "using historical data."
)


print(
    "\nModels Used:"
)


print(
    "- Linear Regression"
)

print(
    "- Random Forest Regression"
)


print(
    "\nEvaluation Metrics:"
)


print(
    "- Mean Absolute Error (MAE)"
)

print(
    "- Root Mean Squared Error (RMSE)"
)

print(
    "- R2 Score"
)


print(
    "\nBest Model Based on Test Results:"
)


# For regression, lower MAE/RMSE and higher R2 are useful.
# Here we select based on lower RMSE.

if rf_rmse < linear_rmse:

    print(
        "Random Forest Regression"
    )

else:

    print(
        "Linear Regression"
    )


print(
    "\nFuture Forecast Years:",
    future_years.tolist()
)


print(
    "\nFinal forecast:"
)


display(
    future_forecast
)


print("\n" + "=" * 70)

print(
    "TASK 5 COMPLETED SUCCESSFULLY!"
)

print("=" * 70)

Output:
======================================================================
TASK 5 - NETFLIX TREND FORECASTING MODEL
======================================================================

Dataset loaded successfully!
Dataset shape: (8790, 10)

======================================================================
DATASET INFORMATION
======================================================================

First 5 rows:
show_id	type	title	director	country	date_added	release_year	rating	duration	listed_in
0	s1	Movie	Dick Johnson Is Dead	Kirsten Johnson	United States	9/25/2021	2020	PG-13	90 min	Documentaries
1	s3	TV Show	Ganglands	Julien Leclercq	France	9/24/2021	2021	TV-MA	1 Season	Crime TV Shows, International TV Shows, TV Act...
2	s6	TV Show	Midnight Mass	Mike Flanagan	United States	9/24/2021	2021	TV-MA	1 Season	TV Dramas, TV Horror, TV Mysteries
3	s14	Movie	Confessions of an Invisible Girl	Bruno Garotti	Brazil	9/22/2021	2021	TV-PG	91 min	Children & Family Movies, Comedies
4	s8	Movie	Sankofa	Haile Gerima	United States	9/24/2021	1993	TV-MA	125 min	Dramas, Independent Movies, International Movies

Column names:
['show_id', 'type', 'title', 'director', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in']

Missing values:
show_id         0
type            0
title           0
director        0
country         0
date_added      0
release_year    0
rating          0
duration        0
listed_in       0
dtype: int64

Duplicate rows:
0

After cleaning:
Dataset shape: (8790, 10)

======================================================================
TIME-BASED FEATURES CREATED
======================================================================
- added_year
- added_month
- added_quarter
- added_day
- added_dayofweek

======================================================================
YEARLY NETFLIX CONTENT ADDITIONS
======================================================================
added_year	content_count
0	2008	2
1	2009	2
2	2010	1
3	2011	13
4	2012	3
5	2013	11
6	2014	24
7	2015	82
8	2016	426
9	2017	1185
10	2018	1648
11	2019	2016
12	2020	1879
13	2021	1498


======================================================================
MONTHLY NETFLIX CONTENT ADDITIONS
======================================================================
added_month
1     737
2     562
3     741
4     763
5     632
6     728
7     827
8     754
9     769
10    760
11    705
12    812
dtype: int64


======================================================================
MOVIE VS TV SHOW YEARLY TREND
======================================================================
type	Movie	TV Show
added_year		
2008	1	1
2009	2	0
2010	1	0
2011	13	0
2012	3	0
2013	6	5
2014	19	5
2015	56	26
2016	251	175
2017	836	349
2018	1237	411
2019	1424	592
2020	1284	595
2021	993	505


======================================================================
FORECASTING DATA
======================================================================
added_year	content_count	time_index
0	2008	2	0
1	2009	2	1
2	2010	1	2
3	2011	13	3
4	2012	3	4
5	2013	11	5
6	2014	24	6
7	2015	82	7
8	2016	426	8
9	2017	1185	9
10	2018	1648	10
11	2019	2016	11
12	2020	1879	12
13	2021	1498	13

======================================================================
TIME-BASED TRAIN-TEST SPLIT
======================================================================
Training years: 2008 to 2018
Testing years: 2019 to 2021

======================================================================
TRAINING LINEAR REGRESSION
======================================================================
Linear Regression MAE: 572.74
Linear Regression RMSE: 657.78
Linear Regression R2: -8.0087

======================================================================
TRAINING RANDOM FOREST REGRESSION
======================================================================
Random Forest MAE: 389.9
Random Forest RMSE: 447.27
Random Forest R2: -3.1653

======================================================================
FORECASTING MODEL COMPARISON
======================================================================
Model	MAE	RMSE	R2_Score
0	Linear Regression	572.7394	657.7779	-8.0087
1	Random Forest Regression	389.9017	447.2712	-3.1653

======================================================================
ACTUAL VS PREDICTED CONTENT
======================================================================
Year	Actual_Content_Count	Linear_Regression_Prediction	Random_Forest_Prediction
0	2019	2016	1094.0	1408.0
1	2020	1879	1225.0	1408.0
2	2021	1498	1356.0	1408.0


======================================================================
TRAINING FINAL FORECASTING MODEL
======================================================================
Final forecasting model trained!

======================================================================
FUTURE NETFLIX CONTENT FORECAST
======================================================================
Year	Predicted_Content_Additions
0	2022	1625
1	2023	1625
2	2024	1625
3	2025	1625
4	2026	1625


======================================================================
FORECAST TREND ANALYSIS
======================================================================
First forecast year: 2022
Last forecast year: 2026
Predicted content in first forecast year: 1625
Predicted content in last forecast year: 1625
Forecast change: 0.0 %

======================================================================
HISTORICAL MONTHLY CONTENT DATA
======================================================================
added_year	added_month	content_count
0	2008	1	1
1	2008	2	1
2	2009	5	1
3	2009	11	1
4	2010	11	1
5	2011	5	1
6	2011	9	1
7	2011	10	11
8	2012	2	1
9	2012	11	1
10	2012	12	1
11	2013	3	1
12	2013	8	1
13	2013	9	2
14	2013	10	3
15	2013	11	2
16	2013	12	2
17	2014	1	2
18	2014	2	2
19	2014	4	2

======================================================================
CONTENT TYPE TREND
======================================================================
Total Movies: 6126
Total TV Shows: 2664


======================================================================
FILES SAVED SUCCESSFULLY
======================================================================
1. netflix_forecasting_model_results.csv
2. netflix_yearly_content_trend.csv
3. netflix_future_content_forecast.csv
4. netflix_forecasting_predictions.csv
5. netflix_trend_forecasting_model.pkl


======================================================================
TASK 5 - PROJECT SUMMARY
======================================================================
Project: Netflix Trend Forecasting Model

Objective:
Forecast future Netflix content release trends using historical data.

Models Used:
- Linear Regression
- Random Forest Regression

Evaluation Metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R2 Score

Best Model Based on Test Results:
Random Forest Regression

Future Forecast Years: [2022, 2023, 2024, 2025, 2026]

Final forecast:
Year	Predicted_Content_Additions
0	2022	1625
1	2023	1625
2	2024	1625
3	2025	1625
4	2026	1625

======================================================================
TASK 5 COMPLETED SUCCESSFULLY!
======================================================================
