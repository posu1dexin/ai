import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('AB_NYC_2019.csv')

# Handle missing values (if any)
data.dropna(inplace=True)

# Convert 'last_review' to DateTime
data['last_review'] = pd.to_datetime(data['last_review'])

# Calculate the number of days since the last review
data['days_since_last_review'] = (pd.to_datetime('today') - data['last_review']).dt.days

data.drop('last_review', axis=1, inplace=True)

# Include 'neighborhood' in the set of features
X = data.drop(['id', 'name', 'host_id', 'host_name', 'latitude', 'longitude'], axis=1)

# Encode categorical variables, including 'neighborhood'
categorical_cols = ['neighbourhood_group', 'room_type', 'neighbourhood']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Split dataset into features and target variable
y = data['price']


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iterate over columns in X_train and check data types
for column in X_train.columns:
    dtype = X_train[column].dtype
    if dtype == 'object':
        print(f"Column '{column}' contains non-numeric values.")


# Linear Regression
linear_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
linear_reg.fit(X_train, y_train)
linear_pred = linear_reg.predict(X_test)

# Polynomial Regression
poly_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2)),
    ('regressor', LinearRegression())
])
poly_reg.fit(X_train, y_train)
poly_pred = poly_reg.predict(X_test)

# Evaluate models
models = {"Linear Regression": linear_pred, "Polynomial Regression": poly_pred}

# Calculate evaluation metrics for each model
evaluation_metrics = {}
for name, prediction in models.items():
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)
    evaluation_metrics[name] = {'MAE': mae, 'MSE': mse, 'R-squared': r2}

# Print evaluation metrics
for name, metrics in evaluation_metrics.items():
    print(f"{name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print()

# Visualize predictions versus actual prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, linear_pred, color='blue', label='Linear Regression')
plt.scatter(y_test, poly_pred, color='red', label='Polynomial Regression')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predictions vs Actual Prices')
plt.legend()
plt.show()
