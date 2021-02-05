# Home Credit Default Risk

The data from this project was collected from [this](https://www.kaggle.com/c/home-credit-default-risk) kaggle competition. See the Jupyter Notebook for the work done to solve this problem.

## Summary

Data Science can play a huge role in Risk Management. This project shows how to use approximately 60 million rows of historical transactional data from multiple data sources in order to predict the default risk of a home credit loan applicant. Engineering of over 600 features, gradient boosted tree model XGBoost, and Bayesian Hyperparameter Optimization were used which resulted in an Area Under the ROC Curve (AUC) score of 0.795. This is an  increase of 0.051 over traditional Logistic Regression.

## Discussion

This project was done in 4 weeks and with the complexity and amount of data this problem presented, XGBoost was chosen to be used because of how robust it is to things like missing values, outliers, and unbalanced datasets. This provides a relatively "clean" start to complex problems that yields decent results especially when given a short amount of time. As a decision tree based model, this also gives interpretability to the features being used having the ability to measure the feature importance the model uses to split the data while building its trees.

It should also be noted that GPU training was utilized for this project which moved the bottle neck of the project from modeling and hyperparameter tuning (which could take up to 10x longer) to the ETL process (extract, transform, and load).

## Results and Conclusion

Overall, this project shows promise with the approach taken with feature engineering and choice of the model. It showed significant increases in AUC from 0.744 to 0.795 when adding more features (631) from other data sources and also using a more complex model over logistic regression. What this translates to is the user of this model has more accurate predictions given over any threshold used for the probability of an client to default on their home loan. For example if the threshold is held at 0.5, then the XGBoost would outperform Logistic Regression in its predictions and would give less false positives which would lead to better decisions when analyzing the risk of each applicant. Additionally, the model returns feature importance which could be used in the future approval process of each loan. The following will list the top 5 features engineered in this project.

### Top Features Engineered

`EXT_SOURCES_MEAN` - The mean of the three different credit scores combined.
 
`PREV_REFUSED_APP_CREDIT_RATIO_MIN` - Previous applications from the client that were refused and the minimum amount requested for those loans.

`PREV_APPROVED_APP_CREDIT_DIFF_MAX` - Previous applications from the client that were approved and the maximum of the difference of amount requested to the amount credit given.

`PREV_LATE_DAYS_DECISION_MEAN` - Previous applications that had late payments in their installments and the mean of the number of days it took to approve the loan.

`LAST_LOAN_PAID_OVER_AMOUNT_MEAN` - The mean of combined installment payments that the client paid over the minimum amount.

## Recommendations

This dataset is very rich in content and has limitless potential for feature engineering. More features could be explored and use of time series techniques could possibly be explored for monthly balance data like credit card balance and installment payments to find trends and predictions for missing current data. Clustering techniques could also be explored to try and group applicants into meaningful groups as a feature which could show improvements in the model. Lastly, while XGBoost is capable of splitting data with missing values, imputation techniques like KNN Imputer could be explored to see if imputing the data would show improvments in AUC.
