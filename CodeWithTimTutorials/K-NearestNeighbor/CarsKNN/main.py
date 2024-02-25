import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas
import numpy
from sklearn import linear_model, preprocessing

data = pandas.read_csv("./car.data")

preprocessor = preprocessing.LabelEncoder()
buying = preprocessor.fit_transform(list(data["buying"]))
maint = preprocessor.fit_transform(list(data["maint"]))
door = preprocessor.fit_transform(list(data["door"]))
persons = preprocessor.fit_transform(list(data["persons"]))
lug_boot = preprocessor.fit_transform(list(data["lug_boot"]))
safety = preprocessor.fit_transform(list(data["safety"]))
cls = preprocessor.fit_transform(list(data["class"]))

predict = "class"

attributes = list(zip(buying, maint, door, persons, lug_boot, safety))
predictions = list(cls)

train_data, test_data, train_answers, test_answers = train_test_split(attributes, predictions, train_size=.10) 


