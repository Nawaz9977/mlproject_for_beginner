import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Transform data to include polynomial features
degree = 5
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X_train)

# Create and train the non-linear regression model
model = LinearRegression()
model.fit(X_poly, y_train)

# Generate data points for visualization
X_plot = np.linspace(0, 5, 100)[:, np.newaxis]
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

# Plot the original data and the regression curve
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Non-Linear Regression Example')
plt.legend()
plt.show()

