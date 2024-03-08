import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplot

keras = tf.keras

data = keras.datasets.fashion_mnist

# special in keras because it makes it easier for us to write all this
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# these label names corresponds to labels in train_labels; index 0 is T-shirt/top, 1 is Trouser.....etc etc
label_names: list[str] = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scale data to smaller number to make it easier for the computer to handle
train_images = train_images/255.5
test_images = test_images/255.5

# keras.Sequential is the sequence of layers of our neural network. left to right 
# This is the architecture of the neural network 
# The input layer is a flatten 28 by 28, 2 dimensional array
# One hidden layer with 128 neurons with activation of RELU
# The output layer with 10 neurons with activation of soft max
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

# compile the model with these parameters 
# TODO: research these parameters
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# epochs is how many times you can see the same image
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"ACC:\t {test_acc}")

# predict using test images
prediction = model.predict(test_images)

for i in range(5):
    pyplot.grid(false)
    pyplot.imshow(test_images[i], cmap=pyplot.cm.binary)
    pyplot.xlabel(f"Actual: {label_names[test_labels[i]]}")
    pyplot.title(f"Model prediction: {label_names[np.argmax(prediction[i])]}")
    pyplot.show()

# # print the label with the highest number from the prediction
# print(label_names[np.argmax(prediction[0])])