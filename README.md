# Ex.No: 6 HOLT WINTERS METHOD

## AIM:
To implement the Holt Winters Method Model using Python.

## ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model predictions against test data
6. Create teh final model and predict future data and plot it

## PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose


file_path = 'cardekho.csv'
df = pd.read_csv(file_path)

print("Dataset Columns:", df.columns)
print(df.head())

df['sale_date'] = pd.to_datetime(df['year'], format='%Y')
df.set_index('sale_date', inplace=True)

default_column_name = 'selling_price' if 'selling_price' in df.columns else df.select_dtypes(include=['int64', 'float64']).columns[0]
series = df[default_column_name].astype(float)
print(f"Using column '{default_column_name}' for forecasting.")

scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(series.values.reshape(-1,1)).flatten()
scaled_data = pd.Series(scaled_values, index=series.index)

plt.figure(figsize=(10,5))
plt.plot(scaled_data)
plt.title("Scaled Data Plot")
plt.show()


decomposition = seasonal_decompose(series, model='additive', period=1)  # yearly data
decomposition.plot()
plt.show()

scaled_data_adj = scaled_data + 1  
train_size = int(len(scaled_data_adj) * 0.8)
train_data = scaled_data_adj[:train_size]
test_data = scaled_data_adj[train_size:]

model_hw = ExponentialSmoothing(train_data, trend='add', seasonal=None, seasonal_periods=1).fit()
test_predictions = model_hw.forecast(steps=len(test_data))

ax = train_data.plot(figsize=(10,6))
test_predictions.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title('Visual evaluation')
plt.show()

final_model = ExponentialSmoothing(scaled_data_adj, trend='add', seasonal=None, seasonal_periods=1).fit()
future_steps = 5  
final_predictions = final_model.forecast(steps=future_steps)

ax = scaled_data_adj.plot(figsize=(10,6), title="Final Holt-Winters Forecast")
final_predictions.plot(ax=ax)
ax.legend(["Original Data", "Final Forecast"])
plt.show()

print("Final Forecast Values:")
print(final_predictions)

```

## OUTPUT:

<img width="968" height="552" alt="image" src="https://github.com/user-attachments/assets/e01b7bba-9bf4-4e7b-b21b-929cbb3337de" />

<img width="1172" height="576" alt="image" src="https://github.com/user-attachments/assets/18b5a203-6cff-420b-b67e-4ec42a96afb3" />

<img width="970" height="610" alt="image" src="https://github.com/user-attachments/assets/169a486b-f2fb-4b61-bb60-68bc19e565be" />

<img width="1225" height="630" alt="image" src="https://github.com/user-attachments/assets/fae52e4c-98fe-44be-b2a5-0ac3c733c0b9" />

<img width="1320" height="727" alt="image" src="https://github.com/user-attachments/assets/70483df5-e408-417b-92b3-2218ecf39b57" />

<img width="527" height="72" alt="image" src="https://github.com/user-attachments/assets/1047b5f3-248b-4085-90bd-5dcbaa7d5652" />

## RESULT:

Thus the program run successfully based on the Holt Winters Method model.
