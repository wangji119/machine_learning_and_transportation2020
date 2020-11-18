# Machine Learning and Transportation  
## Project 1: Buston Housing Prediction Based on Machine Learning  
1.Dataset:  
　In total, we used two datasets of Boston housing prediction.  
　The first dataset we used could be download at https://github.com/udacity/machine-learning/tree/master/projects . There were 3 features and 1 target variable 'MEDV', which had 489 data points for each column.
  The second dataset we used could be download at https://github.com/godfanmiao/DIY_ML_Systems_with_Python_2nd_Edition . There were 13 features and 1 target variable 'MEDV',  which had 506 data points for each column.
  We would like to know the infulence of the dataset on the fit results by comparing the accuracy and error of the two datasets under different methods separately.   
2.Method:  
　　i.Data reading and data pre-processing:  
　　After importing packages, we used pd.read_csv() to read datasets, and we used data.head() and data.info() to check whether the data was read correctly.
We supposed X as the feature columns and y as the target variable, and we set the test_size to be 0.2 that meant 80% of the data were used to train and 20% of the data were used to test.Then we used StandardScaler() to standardize the features because the accuracy of the fit results was improved when it was standardized.  
　　ii.Regression:  
　　We have used 7 based methods to fit the datasets: LR, SVR, KNR, DTR, RFR, GBR, MLPR.
The process of regression was simple. First, import packages. Second, use the fit function to train data and the predict function to compare with the test set. At last, use r2_score to calculate the accuracy of the method.
However, not all methods were so simple. In SVR model, we had to use different kinds of kernel function to custom the fit function, such as 'sigmoid', 'poly', 'linear', 'rbf', etc. Then a very strange scene appeared in Dataset1. No matter how I set up the kernel function，the r2 score was always the negative number. I've looked up the information and it indicates that setting the parameters to adjust the model yields a fit that is also more erroneous than a random guess at a mean. It was really a terrible situation, and SVR model did not fit the Dataset1 actually. But once the features were enough, such as Dataset2, this problem would be solved.
Besides, in MLPRegressor model, we had to set the hidden_layer_sizes. It did take a long time to optimize the function, but we haven't found the best one yet because the time was not enough at last. In additon, if max_iter was too little,there was a convergence warning that means the not best fitting, and 9000 was the edge.
After completing all the steps, it was clear that the MAE and MSE which means the errors of Dataset2 was extremely less than the MAE and MSE of Dataset1. It meant that the results depended on the numbers of features heavily.
The comparison of the accuracy of all methods using in Dataset1 and Dataset2 would be shown in graghs.  
3.Drawing and visualization:  
　　We have totally used two graghs to show the results.  
　　i.Bar gragh  
　　The first one was the bar gragh. It recorded the score of prediction using different methods. As we can see, accuracy scores of the Dataset1 could be ranked from highest to lowest as: 
    GBR＞RFR≈KNR＞MLPR＞DTR＞LR＞＞SVR.
    ![image1](https://github.com/fujunpeng/machine_learning_and_transportation_2020_project/blob/main/image/image1.png)  

　　Ranking of the regression predictive power of multiple classical regression models for the "Boston House Price Forecast" problem:  

| Rank |               Regressors                | R-squared |      MSE       |    MAE    |
| :--: | :-------------------------------------: | :-------: | :------------: | :-------: |
|  1   |          RandomForestRegressor          | 0.826670  | 4086790650.00  | 45910.71  |
|  2   |        GradientBoostingRegressor        | 0.824094  | 4147535115.52  | 49318.95  |
|  3   |           KNeighborsRegressor           | 0.811214  | 4451202000.00  | 50442.85  |
|  4   |              MLPRegressor               | 0.783812  | 5097287652.85  | 52776.75  |
|  5   |            LinearRegression             | 0.706385  | 6922885233.55  | 64456.49  |
|  6   |          DecisionTreeRegressor          | 0.685936  | 7405020000.00  | 66428.57  |
|  7   | SupportVectorRegression(Linear  Kernel) | -0.003011 | 23649074830.10 | 122837.13 |
|  8   |  SupportVectorRegression(Poly  Kernel)  | -0.004588 | 23686253724.46 | 123008.90 |
|  9   | SupportVectorRegression(Sigmod  Kernel) | -0.007067 | 23744697050.01 | 123101.19 |
|  10  |  SupportVectorRegression(RBF  Kernel)   | -0.007830 | 23762683915.53 | 123150.97 |

    While accuracy scores of the Dataset2 could be ranked from highest to lowest as:  
    GBR＞KNR＞RFR＞LR＞MLPR≈SVRpoly＞SVRlinear＞SVRrbf＞SVRsigmoid＞DTR.  
![image2](https://github.com/fujunpeng/machine_learning_and_transportation_2020_project/blob/main/image/image2.png)  

　　Ranking of the regression predictive power of multiple classical regression models for the "Boston House Price Forecast" problem:  

| Rank |               Regressors                | R-squared |    MSE    |   MAE    |
| :--: | :-------------------------------------: | :-------: | :-------: | :------: |
|  1   |        GradientBoostingRegressor        | 0.860479  | 11.905609 | 2.382734 |
|  2   |           KNeighborsRegressor           | 0.794837  | 17.506945 | 2.646863 |
|  3   |            LinearRegression             | 0.768270  | 19.773992 | 3.369037 |
|  4   |  SupportVectorRegression(Poly  Kernel)  | 0.750436  | 21.295807 | 3.232939 |
|  5   |              MLPRegressor               | 0.750210  | 21.315068 | 3.772093 |
|  6   | SupportVectorRegression(Linear  Kernel) | 0.735268  | 22.590114 | 3.276773 |
|  7   |  SupportVectorRegression(RBF  Kernel)   | 0.631065  | 31.481933 | 3.344833 |
|  8   |          RandomForestRegressor          | 0.607798  | 33.467393 | 3.330882 |
|  9   | SupportVectorRegression(Sigmod  Kernel) | 0.579997  | 35.839710 | 4.188697 |
|  10  |          DecisionTreeRegressor          | 0.511331  | 41.699118 | 3.561765 |

　　ii.Comparison chart  
　　The second one was the comparison chart based on the prediction and the test set. From the charts, we could intuitively noticed those good results.  
　　The comparison between the predicted and actual values based on Dataset1 is as follows：  
![image3](https://raw.githubusercontent.com/fujunpeng/machine_learning_and_transportation_2020_project/main/image/image3.png)  
　　The comparison between the predicted and actual values based on Dataset2 is as follows：  
![image4](https://raw.githubusercontent.com/fujunpeng/machine_learning_and_transportation_2020_project/main/image/image4.png)  

　　By the way, we ran into a problem when we were drawing these comparison chart. We found that using only the subplot() function would make each image very small and the spacing between them very small. We made the overall image larger by adding the 'figsize' parameter to the initial plt.figure() function, and added the plt.subplots_adjust() function after several subplot() functions to adjust the spacing between the images to the appropriate scale.