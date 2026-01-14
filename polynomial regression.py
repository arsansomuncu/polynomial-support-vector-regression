# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



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


# Split Data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)


# Feature Scaling
# important for SVR and helpful for polynomial regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Polynomial Regression Implementation
degrees = [1,2,3,4,5,6]
poly_results = []

# create polynomial features
for d in degrees:
  poly = PolynomialFeatures(degree = d)
  X_train_poly = poly.fit_transform(X_train)
  X_test_poly = poly.transform(X_test)

# train the model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# make prediction
y_pred = model.predict(X_test_poly)

# metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

poly_results.append({'degree': d, 'MSE': mse, 'RMSE': rmse, 'R2': r2})
print(f"Degree: {d} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

# plot degree vs. rmse
degrees_list = [res['degree'] for res in poly_results]
rmse_list = [res['RMSE'] for res in poly_results]

plt.figure(figsize=(10, 6))
plt.plot(degrees_list, rmse_list, marker='o', linestyle='-', color='r')
plt.title('Polynomial Degree vs. RMSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.grid(True)
plt.show()




