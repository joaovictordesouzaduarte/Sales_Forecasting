![Rossmann](https://user-images.githubusercontent.com/81034654/120236058-cb01a200-c231-11eb-8c27-48e46658765a.jpg)

<h1> 1) Business problem </h1>
The stores of the Rossmann drugstore chain need to be restored and the CEO needs to decide how much is going to be dedicated to the restoration of each one. The Chief Financial Officer (CFO) requested forecast models for the Analytics team to support his decision that how much budget in each store needs to be allocated.


<h1> 2) Business results </h1>
The gross expected income of a majority of Rossman's stores found was: at the Best Scenario:  R$287,845,044.97; Worst Scenario: R$286,117,462.33 and our Predicted Scenario:  R$286,981,248.00, as you can see in the imagem below. These scenarios are predicted using the mean absolute percentage error.  The same statistical error is applied to each store, individually. 

![image](https://user-images.githubusercontent.com/81034654/120239574-f639c080-c234-11eb-964d-1e30c76c20fa.png)

<h1> 3) Business Hypothesis </h1>
  Through this project, a vary of hypothesis was come up by me. All of these hypotheses are listed in step 02 called "Feature Engineering". New features were created to make     possible a more thorough analysis.

  
  ![image](https://user-images.githubusercontent.com/81034654/120240202-5f6e0380-c236-11eb-904d-2f636f95dad5.png)
  
  
<h1> 4) Solution Strategies </h1>

The strategy adopted was the following:

 Step 01. Data Description: I searched for NAs, checked data types (and adapted some of them for analysis) and presented a statistical description.

Step 02. Feature Engineering: New features were created to make possible a more thorough analysis.

Step 03. Data Filtering: Entries containing no information or containing information which does not match the scope of the project were filtered out.

Step 04. Exploratory Data Analysis: I performed univariate, bivariate and multivariate data analysis, obtaining statistical properties of each of them, correlations and testing hypothesis (the most important of them are detailed in the following section).

Step 05. Data Preparation: Numerical data was rescaled, categorical data was transformed and cyclic data (such as months, weeks and days) was transformed using mathematical trigonometrical functions.

Step 06. Feature selection: The statistically most relevant features were selected using the Boruta package.

Step 07. Machine learning modelling: Some machine learning models were trained. The one that presented best results after cross-validation went through a further stage of hyperparameter fine tunning to optimize the model's generalizability.

Step 08. Model-to-business: The models performance is converted into business values.

Step 09. Deploy Model to Production: The model is deployed on a cloud environment to make possible that other stakeholders and services access its results.


<h1> 5) Machine Learning Performance </h1>
The performance of every trained model, after cross-validation. The columns correspond to the metrics: Mean Absolute Error, Mean Absolute Percentage Error and Root Mean Squared Error.


![image](https://user-images.githubusercontent.com/81034654/120242497-5e8ba080-c23b-11eb-93ea-4c14cd77487b.png)


<h1> 6) Conclusion </h1>

  As we can see, these models have a wide RMSE indicating that this phenomenon studied cannot be modeled by using linear machine learning models. However, the sales forecast and the generated insights provide the CEO with valuable tools to decide the amount of budget that is going to be dedicated to the restoration of each store.
  
  
  
