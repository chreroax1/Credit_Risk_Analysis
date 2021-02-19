# Credit-Risk

## Background
Using the credit card credit dataset from LendingClub I oversampled the data using the RandomOverSampler and SMOTE algorithms, and undersampled the data using the ClusterCentroids algorithm. Then, I used a combination approach of over- and undersampling using the SMOTEENN algorithm. Next, I compared two new machine learning models that reduced bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once finished, I evaluated the performance of these models and created a written recommendation on whether they should be used to predict credit risk.

### Objectives
The goals of this challenge are to:

* Implement machine learning models.
* Use resampling to attempt to address class imbalance.
* Evaluate the performance of machine learning models.

## Resources
* Python 3.7
* Jupyter Notebook
* Libraries:  pandas, numpy, pathlib, collections, scikit-learn, imblearn
* Data: LoanStats_2019Q1.csv (data provided from LendingClub)

## Analysis
The .csv data is read into a pandas DataFrame where it is cleaned and prepared (e.g. textual data converted to numeric).  It is then split into features (X) and target (y) variables as well as training and testing data.  We then set up four different machine learning models where the data is fit.  After each model we assess balanced accuracy score, precision, recall, and f1 score.  

## Results

### Balanced Accuracy Scores
* Naive Random Oversampling:  0.650
* SMOTE Oversampling:  0.662
* Cluster Centroids Undersampling:  0.547
* Combination (Over/Under) Sampling:  0.677

The 2 oversampling methods seem to have similar results.  The accuracy of the undersampling method is much lower than the other methods.  Finally, the combination sampling method shows a slight improvement over the other methods.    

### Final Recommendation
It is important that we correctly predict high risk loans so that we can avoid defaults.  Thus, it would be best to consider using the model that maximizes recall for the high risk category (classification 0).  In this case, the Combination Sampling (SMOTEENN) model performs the best out of the four.  However, by using this model we would incur many false positives (low risk loans classified as high risk) since the precision is low.  There is an opportunity cost of lost profit by not making more of these low risk loans.  Perhaps this model could be used to rule out loans deemed high risk and run another model to further refine the low risk loans.  