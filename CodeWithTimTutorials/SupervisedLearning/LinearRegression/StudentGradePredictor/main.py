import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

def main():
    data = pandas.read_csv("./student-mat.csv", sep=";", index_col=0)
    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
    labelToPredict = "G3"

    attributes = numpy.array(data.drop("G3", axis=1))
    predictions = numpy.array(data[labelToPredict])

    # slices the data into parts so we can separate training and testing data and answers
    # useful for separating training data to testing data 
    training_data_train, training_data_test, actual_answers_train, actual_answers_test = train_test_split(attributes, predictions, train_size=0.25)

    # create a linear regression model and train a best fit line for it using training data and actual answers for training    
    linear = linear_model.LinearRegression()
    linear.fit(training_data_train, actual_answers_train)

    # load pickle studentModel model 
    pickle_in = open("studentModel.pickle", "rb")
    saved_model : linear_model.LinearRegression = pickle.load(pickle_in)

    # this saves the model if the accuracy of the new model is higher
    acc = 0
    current_model_acc =linear.score(training_data_test, actual_answers_test) 
    saved_model_acc = saved_model.score(training_data_test, actual_answers_test) 
    if(current_model_acc > saved_model_acc):
        with open("./studentModel.pickle", "wb") as saved_model:
            pickle.dump(linear, saved_model) 
            acc = current_model_acc
            print(f"current model has better accuracy: {current_model_acc}")
            print(f"saved new model")
    else:
        acc = saved_model_acc
        print(f"saved model has better accuracy: {saved_model_acc}")
        print(f"new model not saved")

    # predict using training data and saved model
    pickle_in = open("studentModel.pickle", "rb")
    saved_model: linear_model.LinearRegression = pickle.load(pickle_in)
    predictions = saved_model.predict(training_data_test)

    # print prediction, data, and actual answers
    for i in range(len(predictions)):
        print(predictions[i], training_data_test[i], actual_answers_test[i])
    print(f"current accuracy: {acc}")

    # Plotting the results using matplotlib
    style.use("ggplot")
    pyplot.scatter(data["G1"], data[labelToPredict])
    pyplot.xlabel = "G1"
    pyplot.ylabel = "Final Grade"
    pyplot.show()

if __name__ == "__main__":
    main()
