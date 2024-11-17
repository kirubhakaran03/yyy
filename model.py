import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load the data
df = pd.read_csv(r'C:\Users\water\Desktop\resume\jamboo.csv')
print(df.head())

# Select features (X) and target (y)
x = df.iloc[:, 0:8]  # Selecting the first 8 columns as features
print(x)
y = df.iloc[:, 8]    # Selecting the 9th column as the target
print(y)

# Apply Min-Max Scaling to features only
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(x)  # Scale only feature columns

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Calculate accuracy metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R-squared (R²): {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")

# Save the model
with open('linear_regression_model3.pkl', 'wb') as file:
    pickle.dump(model, file)
