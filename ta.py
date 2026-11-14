
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("drive/My Drive/ML_Process_Data_Files/Section_12_Model_Evaluation/housing_data.csv")

df.head()

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated']

target = "price"

from sklearn.model_selection import train_test_split

y = df[target]
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

"""## Model

To keep things simple, we'll use linear regression:
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
y_preds = model.predict(X_test)

"""## Evaluation Metrics

The most common evaluation metrics are r2_score, rmse and mae. Here, sklearn has its own implementation:
"""

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

r2 = r2_score(y_test, y_preds)
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
mae = mean_absolute_error(y_test, y_preds)

print("r2_score: {0}".format(r2))
print("rmse: {0}".format(rmse))
print("mae: {0}".format(mae))