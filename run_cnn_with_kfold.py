import keras
from keras.models import Sequential
from keras import layers

import pandas
import numpy

from sklearn.model_selection import KFold
from sklearn import metrics

dataset = pandas.read_csv("dataset.csv")
dataset = dataset.sample(frac=1)

target = dataset.iloc[:,-1].values
data = dataset.iloc[:,:-1].values
data = data/255.0 

##dataset is flattened - we need to transform it back to a matrix
data = data.reshape(-1, 28, 28, 1)
## dimension of the matrix (42000 28x28 matrices with one number in each entry)

# print(target)
# print(data)

split_number = 4
kfold_object = KFold(n_splits=split_number)
kfold_object.get_n_splits(data)

results_accuracy = []
results_confusion_matrix = []

for training_index, test_index in kfold_object.split(data):
	data_training = data[training_index]
	target_training = data[training_index]
	data_test = data[test_index]
	target_test = target[test_index]

	machine = Sequential()
	## (# of nodes (arbitrary), kernel size = dimension of con. matrix (should be odd number so we have a center entry))
	## con matrix determines how many neighbor pixels impact a given pixel (may need larger than 3x3)
	## input shape must match the reshaped matrix dims
	
	machine.add(layers.Conv2D(32, kernel_size = (3,3),
	 			activation = "relu", input_shape=(28,28,1)))
	machine.add(layers.Conv2D(64, kernel_size = (3,3),
				 activation = "relu"))
	
	## grabs the largest entry in each 2x2 matrix 
	## Why? Gets a smaller picture w fewer but more precise info - prevents overfitting
	machine.add(layers.MaxPooling2D(pool_size=(2,2)))

	## model randomly drops 25% of nodes also prevents overfitting can be done before/after flattening
	machine.add(layers.Dropout(0.25))

	## after completing convolution we need to flatten the matrix again
	machine.add(layers.Flatten())

	## Then do dense layers as in ANN
	machine.add(layers.Dense(128, activation = "relu"))	
	machine.add(layers.Dense(64, activation = "relu"))

	machine.add(layers.Dropout(0.25))

	machine.add(layers.Dense(10, activation = "softmax"))

	machine.compile(optimizer = "sgd",
					loss = "sparse_categorical_crossentropy",
					metrics = ['accuracy'])

	# fit data into machine
	machine.fit(data_training, target_training, epochs = 40, batch_size = 128)  

	new_target = numpy.argmax(machine.predict(data_test), axis = -1)

	results_accuracy.append(metrics.accuracy_score(target_test, new_target))
	results_confusion_matrix.append(metrics.confusion_matrix(target_test, new_target))

	print(results_accuracy)
	for i in results_confusion_matrix:
		print(i)

## in general CNN requires more epochs than ANN to converge to max accuracies













