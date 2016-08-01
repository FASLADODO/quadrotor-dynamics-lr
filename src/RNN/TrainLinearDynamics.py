import matplotlib.pyplot as plt
import numpy as np
import time
import os
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import SimpleRNN, Masking, GRU
from keras.models import Sequential
from sklearn.preprocessing import normalize
import keras.optimizers
np.random.seed(1234)


tsteps = 2
batch_size = 21

def process_quadrotor_data():

	path_to_dataset_X_train=os.path.abspath(os.path.join('../..', 'data/LinearDynamicModel/simulatedLinearTrainingInputX.csv'))
	path_to_dataset_y_train=os.path.abspath(os.path.join('../..', 'data/LinearDynamicModel/simulatedLinearTrainingTargetY.csv'))
	path_to_dataset_X_test=os.path.abspath(os.path.join('../..', 'data/LinearDynamicModel/simulatedLinearTestingInputX.csv'))
	path_to_dataset_y_test=os.path.abspath(os.path.join('../..', 'data/LinearDynamicModel/simulatedLinearTestingTargetY.csv'))


	with open(path_to_dataset_X_train)as f:
		X_train = np.loadtxt(open(path_to_dataset_X_train,"rb"),delimiter=",");

	with open(path_to_dataset_y_train)as f:
		y_train = np.loadtxt(open(path_to_dataset_y_train,"rb"), delimiter=",");

	with open(path_to_dataset_X_test)as f:
		X_test = np.loadtxt(open(path_to_dataset_X_test,"rb"),delimiter=",");

	with open(path_to_dataset_y_test)as f:
		y_test = np.loadtxt(open(path_to_dataset_y_test,"rb"), delimiter=",");


	print "dimension of training data", X_train.shape
	print "dimension of test data", X_test.shape


	X_train = np.reshape(X_train, (X_train.shape[0], -1, 1))
	X_test = np.reshape(X_test, (X_test.shape[0], -1, 1))

	print "dimension of training data", X_train.shape
	print "dimension of y test", y_test.shape
	print "dimension of x test", X_test.shape


	return [X_train, y_train, X_test, y_test]

def build_model(X_train, y_train):
	model = Sequential()


	model.add(SimpleRNN(
				output_dim =  50,
				batch_input_shape=(batch_size, tsteps, 1),
				return_sequences=True,
				stateful = True
				))

	model.add(SimpleRNN(
				output_dim = 30,
				batch_input_shape=(batch_size, tsteps, 1),
				return_sequences=False,
				stateful=True
				))

	model.add(Dense(1))

	rmsprop = keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08)

	start = time.time()
	model.compile(loss="mse", optimizer="adagrad", metrics=['accuracy'])
	print "Compilation Time : ", time.time() - start
	return model


def run_net():
	start_timer = time.time()
	epochs = 30
	X_train, y_train , X_test, y_test = process_quadrotor_data()

	model = build_model(X_train, y_train)
	predictions = []
	scores = []

	try:
		model.fit(
			X_train, y_train,
			batch_size = batch_size,
			nb_epoch = epochs,validation_split=0.2)


        	predicted = model.predict(X_test, batch_size=batch_size)
        	predicted = np.reshape(predicted, (predicted.size,))
        	print "Predicted vals ", predicted

        	# Evaluation
        	score = model.evaluate(X_test, y_test, batch_size=batch_size )
	        print "Score: ", score

	except KeyboardInterrupt:
		print "Exception occured"

	try:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(y_test[:(20)])
		plt.plot(predicted[:20])
		plt.show()
	except Exception as e:
		print str(e)
	print 'Training duration (s) :', time.time() - start_timer


if __name__ == "__main__":
   #process_quadrotor_data()
   run_net()

