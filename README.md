
# **Machine Learning and Transportation** 
## **Project 1: Buston Housing Prediction Based on Machine Learning**


1.  Dataset: 

    In total, we used two datasets of Boston housing prediction. 
    The first dataset we used could be download at https://github.com/udacity/machine-learning/tree/master/projects . There were 3 features and 1 target variable 'MEDV', which had 489 data points for each column.
    The second dataset we used could be download at https://github.com/godfanmiao/DIY_ML_Systems_with_Python_2nd_Edition . There were 13 features and 1 target variable 'MEDV', which had 506 data points for each column.
    We would like to know the infulence of the dataset on the fit results by comparing the accuracy and error of the two datasets under different methods separately. 


2.  Method:

    2.1 Data reading and data pre-processing:
    After importing packages, we used pd.read_csv() to read datasets, and we used data.head() and data.info() to check whether the data was read correctly.
    We supposed X as the feature columns and y as the target variable, and we set the test_size to be 0.2 that meant 80% of the data were used to train and 20% of the data were used to test.
    Then we used StandardScaler() to standardize the features because the accuracy of the fit results was improved when it was standardized.

    2.2 Regression:
    We have used 7 based methods to fit the datasets: LR, SVR, KNR, DTR, RFR, GBR, MLPR.
    The process of regression was simple. First, import packages. Second, use the fit function to train data and the predict function to compare with the test set. At last, use r2_score to calculate the accuracy of the method.
    However, not all methods were so simple. In SVR model, we had to use different kinds of kernel function to custom the fit function, such as 'sigmoid', 'poly', 'linear', 'rbf', etc. Then a very strange scene appeared in Dataset1. No matter how I set up the kernel function，the r2 score was always the negative number. I've looked up the information and it indicates that setting the parameters to adjust the model yields a fit that is also more erroneous than a random guess at a mean. It was really a terrible situation, and SVR model did not fit the Dataset1 actually. But once the features were enough, such as Dataset2, this problem would be solved.
    Besides, in MLPRegressor model, we had to set the hidden_layer_sizes. It did take a long time to optimize the function, but we haven't found the best one yet because the time was not enough at last. In additon, if max_iter was too little,there was a convergence warning that means the not best fitting, and 9000 was the edge.
    After completing all the steps, it was clear that the MAE and MSE which means the errors of Dataset2 was extremely less than the MAE and MSE of Dataset1. It meant that the results depended on the numbers of features heavily.
    The comparison of the accuracy of all methods using in Dataset1 and Dataset2 would be shown in graghs.
---

3.  Drawing and visualization:

    We have totally used two graghs to show the results.

    3.1 Bar gragh
    The first one was the bar gragh. It recorded the score of prediction using different methods. As we can see, accuracy scores of the Dataset1 could be ranked from highest to lowest as: 
    GBR＞RFR≈KNR＞MLPR＞DTR＞LR＞＞SVR.
    While accuracy scores of the Dataset2 could be ranked from highest to lowest as:
    GBR＞KNR＞RFR＞LR＞MLPR≈SVRpoly＞SVRlinear＞SVRrbf＞SVRsigmoid＞DTR.

    3.2 Comparison chart
    The second one was the comparison chart based on the prediction and the test set. From the charts, we could intuitively noticed those good results.
    By the way, we ran into a problem when we were drawing these comparison chart. We found that using only the subplot() function would make each image very small and the spacing between them very small. We made the overall image larger by adding the 'figsize' parameter to the initial plt.figure() function, and added the plt.subplots_adjust() function after several subplot() functions to adjust the spacing between the images to the appropriate scale.