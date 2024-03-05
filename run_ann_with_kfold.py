import keras
from keras.models import Sequential
from keras import layers

import pandas
import numpy

from sklearn.model_selection import KFold
from sklearn import metrics

dataset = pandas.read_csv("dataset.csv")
dataset = dataset.sample(frac=1)

target = dataset.iloc[:,:-1].values
data = dataset.iloc[:,:-1].values
data = data/255.0 

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
	machine.add(layers.Dense(512,
				activation = "relu",
				input_shape = (data_training.shape[1],))) # must = # of columns but this does it automatically
	machine.add(layers,Dense(128, # use a 2^x number
				activation = "relu"))
	machine.add(layers,Dense(128, # use a 2^x number
				activation = "relu"))
	machine.add(layers.Dense(64,
				activation = "relu"))
	machine.add(layers.Dense(10, activation="softmax")) #must = # of possible prediction outcomes

	machine.compile(optimizer = "sgd",
					loss = "sparse_categorical_crossentropy",
					metrics = ['accuracy'])

	# fit data into machine
	machine.fit(data, target, epochs = 30, batch_size = 64)  

	prediction = numpy.argmax(machine.predict(new_data), axis=-1)
	results_accuracy.append(metrics.accuracy_score(target_test, new_target))
	results_confusion_matrix.append(metrics.confusion_matrix(target_test, new_target))

print(results_accuracy)
for i in results_confusion_matrix:
	print(i)

## if non kfold accuracy score is very high but kfold accuracy score is not as high = sign of overfitting
## external validation is almost always lower than internal validation





