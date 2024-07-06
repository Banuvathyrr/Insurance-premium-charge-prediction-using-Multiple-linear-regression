

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
**Linearity**: The relationship between the dependent and independent variables is linear.  
**Independence**: The residuals (errors) are independent.  
**Homoscedasticity**: The residuals have constant variance at every level of the independent variables.  
**Normality**: The residuals are normally distributed.  
**No Multicollinearity**: The independent variables are not highly correlated with each other.  


### Metrics for model evaluation
**R-squared (\( R^2 \))**  
Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.  
**Adjusted R-squared**: Adjusts the value for the number of predictors in the model.
**F-statistic**: Tests the overall significance of the model.  
**P-values**: Indicate the significance of individual predictors. A low p-value (typically < 0.05) suggests that the predictor is significantly associated with the dependent variable.  
**Residual Analysis**: Assessing the residuals to check for violations of the assumptions.

### Multicollinearity  
When independent variables are highly correlated, it can affect the stability and interpretation of the coefficients. This can be detected using Variance Inflation Factor (VIF). A VIF value greater than 10 is typically a sign of high multicollinearity.

### Steps involved in performing insurance charge prediction using Multiple linear regression  
Step 1: Data Preparation  
Step 2: Exploratory Data Analysis (EDA)  
Step 3: Data Preprocessing  
Step 4: Splitting the Data  
Step 5: Building the Model  
Step 6: Model Evaluation  
Step 7: Interpreting the Coefficients  


### 🌟 Multiple Linear Regression Analysis:
#### Individual Correlations:

Examining individual correlations helps understand the strength and direction of the relationship between each predictor and the outcome variable.
For example, the correlation between smoker and charges indicates how insurance charges change with smoking status independently.

Multiple Linear Regression:
MLR takes into account the combined effect of all independent variables.
The coefficients in MLR show the effect of each predictor while holding other predictors constant, which can differ from simple correlations.
Multicollinearity:

In MLR, it is important to check for multicollinearity (high correlation among independent variables), as it can affect the stability and interpretation of the model coefficients.
Overall Model Performance:

While individual correlations are useful, the r2 score from MLR provides a measure of how well the entire set of predictors explains the variance in the dependent variable.

### Insightful Correlations:
💡 **Individual Correlations**:  
**Smoking Status**:  
Correlation: 0.78 with insurance charges.  
Interpretation: A strong positive correlation indicates that smokers are significantly likely to incur higher insurance costs.  
**Age**:  
Correlation: 0.29 with insurance charges.  
Interpretation: A weak positive correlation suggests a minor increase in charges with age.    



### Model Performance:
**R² Score**: 0.800! 💪 This means my model explains 80% of the variance in insurance charges, showcasing its reliability and predictive power.
**Combined Effects**: The MLR model accounts for the combined effects of all predictors (e.g., age, sex, BMI, children, smoking status, region) on insurance charges.

### `Sample Code for Multiple linear regression`
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('../data/insurance.csv')

# One-hot encoding for categorical variables
data = pd.get_dummies(data, drop_first=True)

# Define the features (X) and the target (y)
X = data.drop('charges', axis=1)
y = data['charges']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')

# Print the coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)


