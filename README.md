

<h1 align="center">Insurance premium charge prediction using Multiple linear regression</h1>  

### Introduction  
In multiple linear regression, the regression line (or hyperplane in higher dimensions) is drawn considering all the independent variables combined.

### Multiple Linear Regression

In multiple linear regression, the goal is to model the relationship between one dependent variable Y and multiple independent variables X1, X2, ..., Xn. The model can be represented as:

Y = β0 + β1X1 + β2X2 + ... + βnXn + ε

Here:

- Y is the dependent variable.
- X1, X2, ..., Xn are the independent variables.
- β0 is the intercept.
- β1, β2, ..., βn are the coefficients (slopes) associated with each independent variable.
- ε is the error term.

### Assumptions
Linear regression relies on several key assumptions:

1. Linearity: The relationship between the independent and dependent variables is linear.  
2. Independence: Observations are independent of each other.  
3. Homoscedasticity: The variance of residuals is constant across all levels of the independent variables.  
4. Normality: Residuals are normally distributed

### Metrics for model evaluation
**R-squared (\( R^2 \))**  
The R-squared value measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, where a higher value indicates a better fit.

**Mean Squared Error (MSE)**  
The MSE is the average of the squared differences between the observed and predicted values. It provides a measure of the model's accuracy.

**Adjusted R-squared** 
Adjusted R-squared adjusts the R-squared value based on the number of predictors in the model. It is useful for comparing models with different numbers of independent variables.

### Implementation in Python

#### Linear Regression Example in Python

Here is an example of how to perform linear regression using the `scikit-learn` library in Python:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('data.csv')
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')  




