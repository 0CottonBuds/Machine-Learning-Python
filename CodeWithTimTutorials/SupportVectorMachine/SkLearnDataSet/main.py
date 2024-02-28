import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

data = datasets.load_breast_cancer()

# this print statements is used to print the attributes and the labels we want to predict
# print(data.feature_names)
# print(data.target_names)

attributes = data.data
predictions = data.target

train_data, test_data, train_answers, test_answers = train_test_split(attributes, predictions, train_size=0.25)

classes = ["malignant", "benign"]

classifier = svm.SVC()
classifier.fit(train_data, train_answers)

predictions = classifier.predict(test_data)

acc = metrics.accuracy_score(test_answers, predictions)

print(acc)
