

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

### Summary

Multiple Linear Regression is a powerful tool for understanding the relationship between a dependent variable and multiple independent variables. It relies on several assumptions, and its effectiveness depends on the quality of the data and the appropriateness of the model. Careful evaluation of the model and the data is crucial for accurate predictions and insights.



