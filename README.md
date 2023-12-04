# Customer-Segmentation_K-Mean_Clustering-Algorithms
# Overview
This project focuses on creating a robust customer segmentation model using K-Mean clustering algorithms to gain actionable insights into customer behavior. we aim to group customers based on shared characteristics, allowing for targeted and personalized strategies in marketing, product offerings, and customer engagement.
Sure, here are the general steps to follow to create a customer segmentation model:

# Connecting to Data Source 

This code is used to authorize a Google Cloud BigQuery client with service account credentials. The service account key file located at '/work/datapro-405709-a2628eb0c95a.json' is used for authentication. The credentials are then used to instantiate a BigQuery client object, which allows for programmatic interaction with BigQuery services such as executing SQL queries, loading data into tables, exporting data from tables, and managing datasets and jobs.
## Reading The Dataset

The provided code executes a SQL query using a Google BigQuery client and stores the result in a DataFrame. In more detail:
1. First, the `query` method of the Google BigQuery `client` is used to run the SQL query `select * from 'data pro-405709.Projects.Customer Segmentation'`.
2. The `result` method is then called on the `query` object, this retrieves the result of the query execution.
3. Then, the query result is converted into a pandas DataFrame using the `to_dataframe()` function and stored in the `df` variable. This facilitates further analysis and manipulation of the data.

# 1-Exploratory Data Analysis (EDA): 
 Exploratory Data Analysis (EDA) on  data. EDA would give us an initial understanding and summary statistics of our data. To do that, we'll:

- Get the overview of the dataset (shape, columns, datatypes)
- Retrieve some summary statistics (using the describe command)
- Check the presence of any null values
- Look at the unique values for each column
- Check the distribution of important variables

# 2-Data Cleaning: 
In this step, we  deal with missing values, duplicates, and outliers. The approach here would depend on our EDA. We choose to drop  missing values, and remove duplicates and outliers from the dataset.

To remove the outliers We first calculate the Q1 and Q3 for both the 'Quantity' and 'UnitPrice' columns. Q1 represents the 1st quartile and Q3 represents the 3rd quartile.
We then calculate the interquartile range (IQR), which is the range between Q1 and Q3. With this, we define what values can be considered 'normal': if they lie between (Q1 - 1.5*IQR) and (Q3 + 1.5*IQR), they can be considered 'normal'We remove outliers for each column by keeping only the records that satisfy the filter condition.
We repeat the above process for both the 'Quantity' and 'UnitPrice' columns.



# 3-Feature Engineering: 
This code is performing feature engineering on a DataFrame 'df', which is assumed to be defined earlier in the script. The main steps in this code include:

1. Creating a new feature 'TotalPrice' which is the product of 'Quantity' and 'UnitPrice'. This would represent the total purchase price for each row.

2. Generating new date-based features from the 'InvoiceDate' feature. These features include 'Year', 'Month', 'Day', 'Hour', and 'Minute'. This is done by accessing the relevant attribute of the DateTime object in the 'InvoiceDate' series.

3. Constructing a 'Recency' feature which is the difference between the maximum date in 'InvoiceDate' and each of the dates in the series. This can provide information about how recently a customer made a purchase.

4. Creating a 'Monetary' feature by simply copying the 'TotalPrice' column. 

# 4-Data Preprocessing: 
This code is used for standardizing selected columns in a data frame. Standardization is a pre-processing task that is used to transform the data such that it has a mean of zero and a standard deviation of one. This block of code first imports the necessary library, StandardScaler from sklearn. 

StandardScaler object is then initialized and assigned to variable 'scaler'.

The column 'Recency' from the data frame is converted to seconds using the 'total_seconds' method and then to an integer using 'as type'. The conversion to seconds helps in handling datetime fields. After this, any rows that have null values in the 'Recency' column are dropped.

Then a list of columns to be standardized which include 'Quantity', 'UnitPrice', 'TotalPrice', 'Recency', and 'Monetary' is created.

The fit_transform method of the initialized StandardScaler object 'scaler' is then used on the selected columns from the data frame and the result is stored back to the original data frame. The fit_transform method standardizes the features by removing the mean and scaling to unit variance. As a result, the selected columns in the data frame are standardized.


# 5-Model Building: 
We will be performing customer segmentation. A common method to use for customer segmentation is KMeans clustering. This algorithm separates customers into K number of clusters, where each customer belongs to the cluster with the nearest mean. Before applying KMeans, it is important to know the optimal number of clusters. We'll use the Elbow method for this. Let's compute WCSS (Within Cluster Sum of Squares - it's the sum of the squared distance between each member of the cluster and its centroid) for different values of k, and then plot it. The place where this curve bends or forms an 'elbow' indicates the optimal number of clusters (i.e., k value).

The elbow method has been used to determine the optimal number of clusters for KMeans clustering. This method involves plotting the explained variation as a function of the number of clusters and picking the elbow of the curve as the number of clusters to use.

The x-axis represents the number of clusters and the y-axis is the WCSS (Within Cluster Sum of Squares). We can observe that the 'elbow' is at 2, which is optimal for this case. The reason '2' is chosen is that this is the point after which the inertia starts decreasing linearly. So, we will perform segmentation with 2 clusters.

This code performs a k-means clustering on the input data. It starts by initializing a KMeans object with 2 clusters, using the 'k-means++' method for initialization, and allowing up to 300 iterations for the clustering process. The parameter 'n_init' is set to 10, which means that the k-means clustering will be run 10 times with different centroid seeds, and the final results will be the best output of these runs. 

Then, the fit_predict method is invoked on the 'means' object using the input data 'kmeans_df'. It computes the clusters and assigns each data point in 'kmeans_df' to one of the 2 clusters. The labels of these clusters are stored in the variable 'clusters'. Finally, the 'Cluster' column is added to the original DataFrame 'df' where each row is tagged with its corresponding cluster number.

By  looking at the distribution of the different columns per cluster. We'll do this by grouping the data frame by 'Cluster' and calculating the mean for the columns. 

It could be valuable to visualize the clusters to better understand their characteristics. Considering we have more than two features, we might need to use techniques like PCA (Principal Component Analysis) for dimensionality reduction before effectively plotting our data.

The table  shows the mean of each attribute for the two clusters (indexed as 0 and 1). 

From these means, here are the visible patterns:

# Cluster 0:
  - Customers in this cluster buy a small amount of items compared to customers in Cluster 1.
  - The UnitPrice is slightly higher, suggesting they might prefer more expensive items.
  - Their recency value indicates more recent activity compared to Cluster 1.
  - Their monetary value suggests a lower total spend compared to customers in Cluster 1.

# Cluster 1:
  - Customers in this cluster buy more items than customers in cluster 0.
  - They also choose less expensive items.
  - These consumers have not shopped recently given the negative recency score.
  - The monetary value is higher, indicating they have spent more.

# Principal Component Analysis (PCA):
is a technique used for feature reduction or simplification of the dataset. It is particularly useful when dealing with multi-dimensional data. In the context of our analysis, PCA can be used to visualize our clusters in two dimensions, even though our original data has more than two features. Specifically, we can apply PCA to reduce the three features ('Recency', 'Monetary', 'Frequency') into two new 'principal components'. These two new components are directions in data space along which original data points are highly dispersed and each successive component is perpendicular (uncorrelated) to the last, meaning they capture different types of structures in the data. 

By plotting the two principal components, we can visualize our clusters in a two-dimensional space and see how separate they are. This would help us evaluate the performance of our clustering approach visually.

I have applied PCA to the 'kmeans_df' data frame to create two new columns 'PC1' and 'PC2'. These new columns represent the two principal components of the data. This process reduces the dimensionality of the data from the original number of features down to just two, simplifying visualization.

I've then added a new column 'Cluster' to this DataFrame indicating the corresponding cluster each customer was assigned by our K-means model. visualize the clusters with these two newly created principal components. We'll create a scatter plot for 'PC1' and 'PC2', and color the points by their 'Cluster' label. This will allow us to see how the clusters are separated in the PCA-reduced feature space.

The graph  represents the scatter plot of the two principal components we obtained from PCA. The colors represent the different clusters obtained through KMeans. In this representation, we can visualize how the clusters are separated based on the variance of the data explained by these two principal components. As we can see, the KMeans algorithm has created distinct clusters that are visibly separable in this reduced dimensional space.

# Summary
We loaded a dataset from BigQuery, checked its basic characteristics, cleaned it (dealt with missing values, duplicates, and outliers), and created additional features that could help us with customer segmentation (such as recency, frequency, and monetary value). Then, we standardized the necessary features and applied KMeans clustering to segment our customers. To choose the optimal number of clusters, we used the Elbow method and found that 2 clusters were optimal for our case. Finally, we visualized our clusters in a two-dimensional space using PCA for dimensionality reduction. The visualized clusters show clear separations, indicating a good performance of our model.

# 6-Evaluation: 
Evaluating a clustering model can be challenging because we don't have a ground truth to compare the results with. However, there are evaluation metrics that provide insight into the performance of the model.

One popular metric for evaluating clustering models is the Silhouette Score. This measure gives a higher value for a model where clusters are dense and well separated. It ranges from -1 (worst) to 1 (best). A negative score indicates that the clusters might be overlapping, while values near zero indicate overlapping clusters. Values closer to 1 suggest that samples are assigned to the correct clusters.

# 7-Perform Classification: 
We can perform a classification task where we try to predict the cluster a customer belongs to based on the given features. We don't have labels in this case, but since we have performed clustering and assigned each customer to a cluster, we can use these clusters as labels for the purpose of this task. This approach is known as semi-supervised learning.
Let's first prepare our data for the classification task: we'll split the data into training and testing sets. Then, we'll use a simple classification model like Logistic Regression to predict the clusters of unseen data. 

The dataset has been split into a training set and a test set. The training set has 237,617 samples and the test set has 101,836 samples. Both sets have 3 features: 'Recency', 'Frequency', and 'Monetary'.

We train a logistic regression model using the training data, and then we can use this model to predict the clusters for the testing data. We proceed with the training

The Logistic Regression model has been successfully trained on the training data. We use the trained model to predict the clusters of the testing data and evaluate the performance of our model by comparing these predicted clusters with the actual clusters. For performance metrics, we can use accuracy, precision, recall, or F1-score. We'll use accuracy for this case. 

The accuracy score of our model on the test set is 1.0. This means that our model was able to correctly predict the cluster of each customer in the test set. In other words, our model is performing perfectly on the unseen data, which is a very good result. However, it's quite rare to get an accuracy of 1.0 in real-world scenarios. This might indicate that our data is easily separable, or it may suggest overfitting (where the model learns the training data too well and performs poorly on unseen data). Therefore, it would be important to cross-validate these results and ensure that the model is generalizing well.

In conclusion, we were able to successfully build a logistic regression model and use it to classify customers into different clusters based on their Recency, Frequency, and Monetary values. The final accuracy was perfect, indicating the great performance of the model. But as good practice, do cross-validate this.
