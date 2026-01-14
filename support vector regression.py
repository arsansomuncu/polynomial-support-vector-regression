# Import the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


# Load and Inspect Data
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/insurance.csv')
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum()) # check for missing values


# Encoding for Categorical Variables
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])


# Separate Features and Target
X = df.drop('charges', axis = 1)
y = df['charges']


# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)


# Feature Scaling
# important for SVR and helpful for polynomial regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Kernel Testing, Prediction and Metrics
# kernels to test
kernels = ['linear', 'poly', 'rbf']
svr_results = []

for k in kernels:
  svr_model = SVR(kernel=k)

# train
svr_model.fit(X_train, y_train)

# predict
y_pred_svr = svr_model.predict(X_test)

# metrics
mse = mean_squared_error(y_test, y_pred_svr)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_svr)

svr_results.append({'kernel': k, 'MSE': mse, 'RMSE': rmse, 'R2': r2})
print(f"Kernel: {k} | RMSE: {rmse:.2f} | R2: {r2:.4f}")


