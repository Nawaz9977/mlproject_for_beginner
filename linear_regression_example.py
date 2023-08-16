# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate some random data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

diff=y_pred-y_test
print(y_pred, "-" , y_test, "=" , diff)

# Plot the original data and the regression line
#plt.scatter(X_test, y_test, color='blue', label='Test Data')
#plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
#plt.xlabel('X')
#plt.ylabel('y')
#plt.title('Linear Regression Example')
#plt.legend()
#plt.show()
