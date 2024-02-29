import numpy
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

data = load_digits()

# scale makes large numbers small for more lenient computation overhead
attributes = scale(data.data)
prediction_labels = data.target

# number of centroids by the number of unique prediction labels
k = len(numpy.unique(prediction_labels))

samples, features  = attributes.shape

# this function is used to score the accuracy of the k mean clustering model 
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(prediction_labels,  estimator.labels_),
             metrics.completeness_score(prediction_labels,  estimator.labels_),
             metrics.v_measure_score(prediction_labels,  estimator.labels_),
             metrics.adjusted_rand_score(prediction_labels,  estimator.labels_),
             metrics.adjusted_mutual_info_score(prediction_labels,   estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))
    
classifier = KMeans(n_clusters=k, init='random', n_init=20)

bench_k_means(classifier, "1", attributes)


