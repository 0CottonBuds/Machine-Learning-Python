import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas
import numpy
from sklearn import linear_model, preprocessing

data = pandas.read_csv("./car.data")

preprocessor = preprocessing.LabelEncoder()
# fit transform converts string data to computable int data for the AI model. Do this for non numerical attributes
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

train_data, test_data, train_answers, test_answers = train_test_split(attributes, predictions, train_size=.1) 

model = KNeighborsClassifier(n_neighbors=9)
model.fit(train_data, train_answers)
acc = model.score(test_data, test_answers)

print(acc)

predicted_data = model.predict(test_data)
names = ["unacc", "acc", "good", "vgood"]

for i in range(len(predicted_data)):
    print(f"Data: {test_data[i]}", f"Predicted: {names[predicted_data[i]]}", f"Actual Answer: {names[test_answers[i]]}")
