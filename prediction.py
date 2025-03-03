import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sample Dataset (Indian Shoe Sizes)
data = {'Age': [5, 10, 15, 20, 25, 30, 40, 50, 60],
        'Shoe_Size': [10, 14, 18, 22, 24, 26, 28, 29, 30]}

df = pd.DataFrame(data)

# Splitting Data
X = df[['Age']]  # Features
y = df['Shoe_Size']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}, MSE: {mse}, R2 Score: {r2}')

# Plot
plt.scatter(df['Age'], df['Shoe_Size'], color='blue', label='Actual Data')
plt.plot(df['Age'], model.predict(df[['Age']]), color='red', label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Shoe Size (India)')
plt.legend()
plt.show()
