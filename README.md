# Project Title: Sales Quantity Forecasting for Pizza Dataset 2015

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Objective](#objective)
- [Analysis Approach](#analysis-approach)
- [Key Findings](#key-findings)
- [How to run code](#how-to-run-code)
- [Technologies Used](#technologies-used)
- [Results & Visualizations](#results--visualizations)
- [Recommendation](#recommendation)
- [Contact](#contact)

## Overview

This project focuses on forecasting pizza sales quantity using machine learning techniques. By analyzing transaction data, we aim to predict the total quantity sold weekly for specific pizzas of different sizes in New Jersey. The ultimate goal is to aggregate these weekly predictions to generate an annual sales forecast for the next year.

## Dataset

The analysis is based on the [data_pizza.xlsx](data_pizza.xlsx)
- Time period Covered: 2015
- Number of Record: 21288 rows
- Number of Features: 10
- Key Variables: according to [data_dictionary.xlsx](data_dictionary.xlsx)

## Objectives

The objective is to analyze and model the total quantity sold on a weekly basis using categorical features, forecast total sales and quantity for the upcoming year by aggregating weekly predictions, and identify key factors influencing pizza sales to optimize business strategies accordingly.

## Analysis Approach
1. Exploratory Data Analysis (EDA): Data cleaning, handling missing values, and visualizing trends.
2. Data Preprocessing and Data Selection: Applying transformations such as log transformations and standardization.
3. Model Selection and Evaluation: Comparing multiple supervised learning models.
4. Forecasting Sales and Quantity: Using the best-performing model for predictions.
5. Insights and Recommendations: Deriving actionable insights for business decisions.

To predict the quantity sold based on categorical features, several prediction models can be considered. The choice of model depends on the nature of the data and the specific problem requirements. Here are some suitable models for handling categorical features:

1. Linear Models with Regularization:
- Lasso Regression (L1 regularization): Helps in feature selection by shrinking less important feature coefficients to zero.
- Ridge Regression (L2 regularization): Regularizes model complexity by shrinking coefficients without setting them to zero.

Advantages:
- Good for datasets with many categorical variables, especially if one-hot encoding leads to a high-dimensional space.
- Efficient and interpretable.

2. Decision Tree Regression:
- Handles both numerical and categorical data natively.
- Can capture non-linear relationships and interactions between features.

Advantages:
- Intuitive and easy to interpret.
- No need for extensive preprocessing of categorical data.

3. Random Forest Regression:
- An ensemble of decision trees, which improves generalization by reducing variance.
- Handles categorical features well and is robust to overfitting.

Advantages:
- High accuracy and can capture complex patterns.
- Feature importance can be derived, providing insights into the impact of different features.

4. Gradient Boosting Machines (GBM): Handle categorical features effectively.

Advantages:
- Excellent predictive performance.
- Built-in handling for categorical features.

5. Support Vector Machines (SVM):
- Can be used with kernels to capture non-linear relationships.
- Requires preprocessing of categorical data into numerical format (e.g., one-hot encoding).

Advantages:
- Effective for both linear and non-linear data.
- Robust to overfitting in high-dimensional spaces.

## Key Findings
- The Gradient Boosting Machine (GBM) model provided the best forecasting performance.
- Total predicted sales and quantity for 2016 showed a decline compared to 2015.
- Large and Medium pizza sizes were the most popular.
- Non-holiday weeks saw significantly higher sales than holiday weeks.
- Seasonal shifts: Highest sales in summer (2015) and winter (predicted 2016).
- Medium-priced pizzas ($15-$25) dominated sales.
- Weeks 39, 52, and 53 consistently experienced sales drops.

## How to run code

1. Install Required Libraries: Ensure all necessary libraries such as pandas, matplotlib, seaborn, Scikit-learn,... are installed like in the [file](sales_quantity_predicting.ipynb)
2. Load the Dataset: Import the dataset by loading the [file](data_pizza.xlsx)
3. Run the Analysis Notebooks: Execute the analysis notebooks in Jupyter to process the data, build and train the model, and visualize the results.

## Technologies Used

- Python: Data analysis and preprocessing were performed using pandas and numpy. For modeling, various algorithms were explored, including Lasso Regression, Ridge Regression, Decision Tree, Random Forest, Gradient Boosting Machine (GBM), and Support Vector Machines (SVM), etc. Model evaluation and hyperparameter tuning were done using scikit-learn, including GridSearchCV for optimizing model parameters.

- Visualization: Visualizations were created with matplotlib and seaborn to analyze trends and evaluate model performance. Feature importance was also visualized to understand the impact of different features on the predictions.

## Results & Visualizations

### Choosing Dataset and Model

![Screenshot 2025-02-19 145955](https://github.com/user-attachments/assets/f9f04ad2-2fc1-41b9-b267-96e657ba5063)

Figure 1: Report the RMSE for the training and test sets for linear regression and Lasso regression model for each dataset

Finding: Dataset with log transformations and Dataset with log transformations + standardization perform better 

![Screenshot 2025-02-19 150443](https://github.com/user-attachments/assets/989012b3-c950-4add-9063-daf65179a0e8)

Figure 2: Model Validation - Lasso Regression Model 

![Screenshot 2025-02-19 150738](https://github.com/user-attachments/assets/edf0cb56-ede9-4633-9c54-6df4956d08e3)

Figure 3: Model Validation - Ridge Regression Model

![Screenshot 2025-02-19 150841](https://github.com/user-attachments/assets/51679730-1ddc-4e7c-aa09-50e9b240c524)

Figure 4: Model Validation - Decision Tree Regression Model

![Screenshot 2025-02-19 150938](https://github.com/user-attachments/assets/d28fc361-d24e-470e-b022-26fa47f78a6d)

Figure 5: Model Validation - Random Forest Regressor Model 

![Screenshot 2025-02-19 151031](https://github.com/user-attachments/assets/fbf5386d-4ed8-444b-bddb-686447b1ccc8)

Figure 6: Model Validation - Gradient Boosting Machine (GBM) Model

![Screenshot 2025-02-19 151128](https://github.com/user-attachments/assets/6aa0b5ca-b776-4a02-976f-85f190be7fe6)

Figure 7: Model Validation - Support Vector Regression (SVR) Model

![Screenshot 2025-02-19 151230](https://github.com/user-attachments/assets/e919daa0-a972-4474-afdf-4198b17fd238)

Figure 8: Model Validation - MCA  Model

Conclusion: The Gradient Boosting Machine (GBM) is the best model based on the lowest RMSE on both the training and test sets. It indicates a good balance between fitting the training data and generalizing to new, unseen data.

### Result for prediction

2015 total sales: 817.86K

2015 total quantity: 49574

2016 Total Sales (predict): 766410.2264293174

2016 Quantity (predict): 46456.417408155285

Seems like predicted quantity in 2016 according to the predicting model has lower sales and lower quantity than 2015. For further understanding, we can plot total sales, quantity by month, week with the predicted quantity to find out the reasons. 

![image](https://github.com/user-attachments/assets/4067f5c6-7b67-4f04-b76f-5d5635212b77)
Total quantity sold by pizza name in 2015

![image](https://github.com/user-attachments/assets/bc9981e4-f7e3-485c-b6f7-72a7a7becddd)
Total predicted quantity sell by pizza name in 2016

Finding:

Total quantity sold by pizza name in 2015:
- top 6 pizza around 2400-2500 quantity sold: the classic deluxe pizza > the bbq chicken pizza> the hawaiian pizza > the pepperoni pizza > the thai chicken pizza > the california chicken pizza 
- bottom pizza around 500 quantity sold: the bire carre pizza.
- other pizza around nearly 1000-2000 quantity sold.

Total predicted quantity sell by pizza name in 2016:
- top 6 pizza around 2000-2500 quantity sold: the pepperoni pizza > the classic deluxe pizza > the california chicken pizza > the thai chicken pizza > the hawaiian pizza > The Italian Supreme Pizza
- bottom pizza around 500 quantity sold: the bire carre pizza.
- other pizza around nearly 800-2000 quantity sold.

![image](https://github.com/user-attachments/assets/aab5140a-4263-4570-911c-00eabbe2da26)
Visualize the distribution of categories with target variable with Bar Plots in 2015

![image](https://github.com/user-attachments/assets/c22a9bf0-55f3-4506-99d6-2f3dc33346e3)
Visualize the distribution of categories with target variable with Bar Plots in 2016

Finding:

Visualize the distribution of categories with target variable with Bar Plots in 2015:
- quantity sold by size: L>M>S>XL>XXL
- non-holiday sold more than holiday around 2.7 times
- Quantity sold by season: summer>spring>winter>fall (14K>13.5K>11.8K>10.5K)
- Quantity sold by price: medium price around $15-$25> low price below $15 > high price above $25 (30K>15K>500)

Visualize the distribution of categories with target variable with Bar Plots in 2016:
- quantity sold by size: L>M>S>XL>XXL
- non-holiday sold more than holiday around 2.2 times
- Quantity sold by season: winter>spring>fall>summer (14K>12K>11.5K>9.8K)
- Quantity sold by price: medium price around $15-$25> low price below $15 > high price above $25 (30K>14K>500)

![image](https://github.com/user-attachments/assets/b4d4c0b5-5cb7-450f-b968-c313a9b93ecb)
Total Quantity Sold vs. Week of Year (2015)

![image](https://github.com/user-attachments/assets/2aa6ba5d-5e02-49c9-806e-00c3d61399e1)
Predicted Quantity Sell vs. Week of Year (2016)

Finding: 

Total Quantity Sold vs. Week of Year (2015)
- The sales start low at around 600 in Week 1.
- From Week 2 to Week 38, sales increase and stabilize between 900 and 1,050 units.
- Week 39 sees a drop to around 650, but sales recover in Week 40, stabilizing again around 900–1,050 until Week 47.
- Week 48 peaks at 1,200, followed by a decline in the next few weeks, stabilizing around 950 until Week 51.
- Week 52 drops to 650, and Week 53 further declines to around 400.

Predicted Quantity Sell vs. Week of Year (2016)
- The year starts higher than 2015, at around 700 in Week 1.
- From Week 2 to Week 13, sales increase and stabilize around 1,000 units.
- Week 14 sees a drop, with sales stabilizing around 900 until Week 26.
- Week 27 marks another decline, with sales remaining between 700 and 780 until Week 38.
- Week 39 drops to 600, but Week 40 sees a recovery to 800, followed by a gradual increase until Week 48, reaching 1,000 units.
- Week 49 experiences a slight drop to 950, followed by a gradual decline to 900 in Week 51.
- Week 52 drops to 700, and Week 53 further declines to 500.

Below is plot of important features
![image](https://github.com/user-attachments/assets/669a791e-fb53-40f8-9482-ca70ce4a6d2d)
Feature importance from GBM model

Consider features that have significant impact: price, week_of_year

## Recommendation:
- Focus promotions on high-performing pizzas to drive even more sales.
- Since 2016 predicts higher winter sales, consider launching winter-themed promotions (e.g., family bundles, seasonal flavors).
- Identify why summer sales are predicted to decline—adjust marketing efforts accordingly.
- Consider revising pricing strategy for high-priced pizzas—perhaps premium toppings or meal combos could justify the price.
- Adjust Summer Marketing: Address the expected decline in summer sales.

## Contact

📧 Email: pearriperri@gmail.com

🔗 [LinkedIn](https://www.linkedin.com/in/phan-chenh-6a7ba127a/) | Portfolio

