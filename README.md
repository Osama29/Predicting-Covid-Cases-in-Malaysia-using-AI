# Predicting Covid Cases in Malaysia using AI
 
## Design Decision
This laboratory report's design decision tries to anticipate the daily total number of deaths caused by COVID19. The goal of the study is to determine the number of deaths or death rate, particularly when a new epidemic wave or variant emerges. Also, to determine the relationship between vaccination status and death number. Therefore, supervised learning aids in predicting the future death rate due to COVID19, allowing the government to instantly plan for solution to aid in reducing the death number. The selected calendar range is between January 2021 and June 2021. The graph below depicts a line chart of new death cases per day in Malaysia during the epidemic.
![Screenshot 2022-09-13 165011](https://user-images.githubusercontent.com/53341547/189856934-f2f50231-6137-415e-94c9-fc25ddc6aacb.png)
The datasets are compiled by the Malaysian Ministry of Health and retrieved from GitHub. The CSV files, chosen features, and chosen target are as follows:
1. CSV files
* cases_malaysia.csv: daily COVID cases in Malaysia
* vax_malaysia.csv: vaccination status of people in Malaysia
* deaths_malaysia.csv: daily death cases due to COVID in Malaysia
2. Selected features
* date: year, month and day
* cases_unvax: unvaccinated covid patient number
* cases_pvax: partially vaccinated covid patient number
* cases_fvax: fully vaccinated covid patient number
* cases_new: total number of new cases per day
* cases_active: total number of active cases per day
* death_unvax: unvaccinated covid patient death number
* death_pvax: partially vaccinated covid patient death number
* death_fvax: fully vaccinated covid patient death number
* death_boost: fully vaccinated with booster covid patient death number
3. Selected target
* deaths_new: total number of new covid patient death number

## Justification of Feature Selection:
The difference between the actual and predicted results on the machine learning model highly depend on the selected features. The first factor that would affect the result of machine learning model is vaccination status. There are three types of vaccination status: unvaccinated status, partially vaccinated status, and fully vaccinated status. Unvaccinated people would have a higher chance than others to die due to the disease compared to people who are partially vaccinated and fully vaccinated. According to Romo (2021), unvaccinated people are eleven times more likely to die due to the disease. This is because vaccines could effectively protect people even against the delta variety. Moreover, the risk of infection was almost five times lower in those who had received vaccinations. Hence, if people have not been vaccinated, they would have a higher chance of getting an infection and may die due to no protection from the vaccine.

Besides, the second factor affecting the machine learning model is the number of new cases per day. The new cases per day include all the vaccination statuses. Although vaccines would give people powerful protection, vaccines require time to build immunity inside the body of people. For instance, partially vaccinated people need third doses of a vaccine to increase the immunity inside their body. However, it takes a few weeks for the third dose to provide maximum protection after taking the third dose (World Health Organization, 2021). Hence, the new case per day would be affected by this situation. People waiting to build immunity may get an infection due to weak immunity.

Furthermore, the active cases per day might be the third factor contributing to the number of new covid patient deaths. The active cases indicate the number of infected people. The infected people would have a higher chance of dying due to the disease compared to non-infected people. According to Ellis (2021), people would suffer several health risks up to 6 months after infecting the disease. People who are infected would have a higher chance of dying. Hence, the number of active cases per day might be considered during the machine learning model.

## Supervised Learning Techniques
Linear Regression, Polynomial Regression, and Lasso Regression are the supervised learning algorithms chosen to be employed in this study. As predicting the number of deaths is a type of regression problem, regression techniques are used. The final objective of the regression technique is to plot a line or curve that best fits the data. Thus, the models are evaluated using the mean square error of the train and test sets and the line of best fit (perfect regression).

### Linear Regression
Linear regression is used to predict the value of a dependent variable by using the independent variable. The difference between the expected and actual output values minimizes linear regression by fitting a straight line or surface.
#### Implementation
Firstly, copy the required features data for prediction from a table named features into a variable named x. Then, copy the existing new death cases per day for training the model into a variable named y. Next, remove the column about the death cases from the table. This step is described in the code below:

```Python
# Copy data from features into x

x = features.copy()

# Copy deaths_new from features into Y

y = features['deaths_new']

# Deletes the deaths_new column from X

del x['deaths_new']
```

The step below is to create train and test splits of the data sets. Four data sets will be named x_train, x-test, y_train, and y_test. The test size is 0.3.

```Python
#Define training, testing and validation data set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
```

The step below shows that the instance of linear regression is created. Each data set, such as x_train and y_train, fits a basic linear regression model on the training data. Calculate the mean squared error (mse) on both the train and test sets. Mean squared error shows the difference between actual and predicted values.

```Python
#linear Regression

#Create an instance of the class
LR = LinearRegression()

#Fit the instance on the data
LR.fit(x_train, y_train)

#Predict the expected on the data
y_predict = LR.predict(x_test)
print(y_predict)

#Use Linear Regression to visualise
y_train_pred = LR.predict(x_train)
y_train_mse = mean_squared_error(y_train, y_train_pred)
y_test_pred = LR.predict(x_test)
y_test_mse = mean_squared_error(y_test, y_test_pred)
```

The step below shows plotting the linear regression graph using scatter plots.
```Python
#Plotting graph
plt.scatter(y_test, y_predict, color='blue')
plt.title('Linear Regression')
plt.xlabel('Test Result')
plt.ylabel('Predicted Result')
plt.show()

print("Mean Squared Error")
print("Train: ", y_train_mse)
print("Test: ", y_test_mse)
```

The step below is to visualize a more detailed linear regression by putting the best fit lines into it. The p1 and p2 are used to draw the best fit lines.

```Python 
#Final result for linear regression
plt.figure(figsize= (10,10))
plt.scatter(y_test, y_predict, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_predict), max(y_test))
p2 = min(min(y_predict), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('Actual Result', fontsize = 15)
plt.ylabel('Predicted Result', fontsize = 15)
plt.axis('equal')
plt.show()
```
## Polynomial Regression
Polynomial regression is a form of linear regression. To transform linear regression into polynomial regression, certain polynomial terms are added to it owing to the non-linear connection between the dependent and independent variables.

### Implementation
The step below shows that the instance of linear regression is created with the polynomial degree of 2 and does not include bias. The x_train and x_test are fitted and transformed into polynomial form. Each data set, such as x_poly_train and y_train, fits a basic linear regression model on the training data. Calculate the mean squared error (mse) on both the train and test sets. Mean squared error shows the difference between actual and predicted values.

```Python
#Polynomial Regression 

#Create an instance of the class with degree of 2 and no include_bias
poly_transform = PolynomialFeatures(degree= 2, include_bias= False)
x_poly_train = poly_transform.fit_transform(x_train)
x_poly_test = poly_transform.transform(x_test)

#Create an instance of the class for LR and fit the train data
LR = LinearRegression()
LR.fit(x_poly_train, y_train)

#Use LR to visualise
poly_y_train_pred = LR.predict(x_poly_train)
poly_y_train_mse = mean_squared_error(y_train, poly_y_train_pred)
poly_y_test_pred = LR.predict(x_poly_test)
poly_y_test_mse = mean_squared_error(y_test, poly_y_test_pred)
poly_y_test_mse = mean_squared_error(y_test, poly_y_test_pred)
```
The step below shows plotting the polynomial regression graph using scatter plots.

```Python 
#Plotting graph
plt.scatter(y_test, poly_y_test_pred, color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Actual Result')
plt.ylabel('Predicted Result')
plt.show()

print("Mean Squared Error")
print("Train: ", poly_y_train_mse)
print("Test: ", poly_Y_test_mse)
```

The step below is to visualize a more detailed polynomial regression by putting the best fit lines into it. The p1 and p2 are used to draw the best fit lines.

```Python
#Final Result for polynomial regression 
plt.figure(figsize=(10,10))
plt.scatter(y_test, poly_y_test_pred, c = 'crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(poly_y_test_pred), max(y_test))
p2 = min(min(poly_y_test_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.label('Actual Result', fontsize = 15)
plt.ylabel('Predicted Results', fontsize= 15)
plt.axis('equal')
plt.show()

## Lasso Regression
Lasso regression is another form of linear regression that uses shrinkage and performs L1 regularization. Greater penalties provide coefficient values closer to zero, which is excellent for creating more straightforward models.

### Implementation:
The step below shows that the instance of lasso regression is created with the alpha of 3. Each data set, such as x_ train and y_train, fits a basic lasso regression model on the training data. Calculate the mean squared error (mse) on both the train and test sets. Mean squared error shows the difference between actual and predicted values.

```Python
#Lasso Regression

#Create an instance of the class with alpha of 3
LS = Lasso(alpha= 3.0)

#Fit the train data
LS.fit(x_train, y_train)

#USe Lasso to visualise
y_train_pred = LS.predict(x_train)
y_train_mse = mean_squared_error(y_train, y_train_pred)
y_test_pred = LS.predict(x_test)
y_test_mse = mean_squared_error(y_test, y_test_pred)
```

The step below shows plotting the lasso regression graph using scatter plots.
```Python
#Plotting graph
plt.scatter(y_test, y_test_pred, color = 'blue')
plt.title('Lasso Regression')
plt.xlabel("Actual Result')
plt.ylabel('Predicted Result')
plt.show()

print("MEan Squared Error")
print("Train: ", y_train_mse)
print("Test: ", y_test_mse)
```

The step below is to visualize a more detailed lasso regression by putting the best fit lines into it. The p1 and p2 are used to draw the best fit lines.

```Python
#Final result for lasso regression
plt.figure(figsize=(10,10))
plt.scatter(y_test, y_test_pred, c= 'crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_test_pred), max(y_test))
p2 = min(min(y_test_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('Actual Result', fontsize = 15)
plt.ylabel('Predicted Result', fontsize = 15)
plt.axis('equal')
plt.show()
```

## Results & Analysis
The models are evaluated with mean square error (MSE) of train and test sets and perfect regression (best fit) line. The MSE is used to evaluate the quality of a model based on the predictions made on the entire training dataset relative to the actual label/output value. MSE can be used to represent the cost or loss associated with the predictions. The lower the mean square error indicates a better regression model.
Meanwhile for perfect regression line, the closer the data points are to the regression line, the higher the accuracy of the model. The table below shows the evaluation and comparison of Linear Regression, Lasso Regression and Polynomial Regression.

|Supervised Learning Algorithm | Mean Square Error (MSE) | Perfect Regression/ Best Fit Line |
|------------------------------|-------------------------|-----------------------------------|
| Linear Regression | ![image](https://user-images.githubusercontent.com/53341547/189932484-967612af-e728-4860-8bd7-3410f90add3f.png) | ![image](https://user-images.githubusercontent.com/53341547/189932580-a0b9bea4-1c72-40b7-84f8-a4b526eb91e7.png) |
|------------------------------|-------------------------|-----------------------------------|
|Polynomial Regression | ![image](https://user-images.githubusercontent.com/53341547/189932880-1ef9d8d2-98d4-480d-8da4-070a6a95cdcc.png) | ![image](https://user-images.githubusercontent.com/53341547/189932933-c1404504-ddca-461c-aac4-e493e1c32051.png)|
|------------------------------|-------------------------|-----------------------------------|
| Lasso Regression | ![image](https://user-images.githubusercontent.com/53341547/189933246-1c327f13-9390-40ec-bf46-a8ecad28eabf.png) | ![image](https://user-images.githubusercontent.com/53341547/189933318-3c948640-41c1-40a0-aa8d-8835e04117d0.png) |
|------------------------------|-------------------------|-----------------------------------|


