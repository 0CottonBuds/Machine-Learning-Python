K-Means Clustering

This model have a number centroids denoted by k. The model learns by creating centroids (qty denoted by k) on random positions and then setting the points of data to the nearest centroids. After that the centroids are moved based on the average position of all points that is nearest to the centroids. Repeat this over and over. The model is finished learning if the centroids do not move when finding the average position of the points that is nearest to it.

This is a slow model because for each point we need to look for its nearest centroid. 0(i(f*(p*c))) where p is the number of data points, f is the number of features, i is the number of iterations and c is the number of centroids. This model is slower than other machine learning models but based on what tim said on his tutorial it is faster in general for clustering models.

I don't quite understand this model I need to revisit this in the future.