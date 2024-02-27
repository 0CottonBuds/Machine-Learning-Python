import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split 


data = datasets.load_breast_cancer()

print(data.feature_names)
print(data.target_names)

attributes = data.data
predictions = data.target

train_data, test_data, train_answers, test_answers = train_test_split(attributes, predictions, train_size=0.25)

classes = ["malignant", "benign"]
