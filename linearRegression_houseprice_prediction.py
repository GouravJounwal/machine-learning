import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample data

df=pd.read_csv('/areaprice.csv')
# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['SquareFeet']]  # feature must be 2D
y = df['Price']         # target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Display results
for i, (actual, predicted) in enumerate(zip(y_test, y_pred)):
    print(f"Sample {i+1}: Actual: {actual}, Predicted: {predicted:.2f}")

# Plot
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Square Feet')
plt.ylabel('House Price')
plt.title('Linear Regression: House Price Prediction')
plt.legend()
plt.show()
