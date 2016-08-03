from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import os,sys
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K


def load_data():


    path_to_dataset_X=os.path.abspath(os.path.join('../../..', 'data/NonlinearDynamicModel/speedDataNormalized.csv'))
    path_to_dataset_train=os.path.abspath(os.path.join('../../..', 'data/NonlinearDynamicModel/speedDataNormalizedTestData.csv'))

    speedDataNormalized = np.loadtxt(open(path_to_dataset_X,"rb"),delimiter=",")
    speedTestBatchNorm = np.loadtxt(open(path_to_dataset_train, "rb"),delimiter=",")

    x_train = speedDataNormalized
    x_test = speedTestBatchNorm

    print x_train.shape
    print x_test.shape

    return x_train, x_test

def buildModel():

    # Size of encoding
    encoding_dim = 1
    input_dim = 5

    # ---------------------------------------------------------
    # Construct encoder and decoder layers
    # ---------------------------------------------------------
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
    decoded = Dense(input_dim, activation='linear')(encoded)

    # ---------------------------------------------------------
    # Autoencoder - Maps input to its reconstruction
    # ---------------------------------------------------------
    autoencoder = Model(input=input_img, output=decoded)

    # ---------------------------------------------------------
    # Encoder - Maps input to the encoded representation
    # ---------------------------------------------------------
    encoder = Model(input=input_img, output = encoded)

    # ---------------------------------------------------------
    # Decoder : last layer of the autoencoder model
    # ---------------------------------------------------------
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]

    # ---------------------------------------------------------
    # Decoder Model
    # ---------------------------------------------------------
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss = 'mse')

    return autoencoder, encoder, decoder

def train(autoencoder,  epochs):

    autoencoder.fit(x_train, x_train,
                    nb_epoch=epochs,
                    batch_size=10,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    return autoencoder

def predict(autoencoder, encoder, decoder, x_train, x_test):

    # Let's take 450 flight data as training
    # and do prediction
    print "Size of inputs to predict: ", x_train.shape
    print "Size of inputs to predict: ", x_test.shape

    flight_train_size = 497 * 450
    flight_test_size = 497 * 50
    step_size = 5

    f = file('networkTrainingData.csv', 'a')

    for i in range(0,flight_train_size) :
        sequential_input = x_train[i,:]
        #do prediction on this
        #print "sequence grabbed", sequential_input
        reshaped_seq = np.reshape(sequential_input, (1,step_size))
        encoded_flight = encoder.predict(reshaped_seq)
        decoded_flight = decoder.predict(encoded_flight)


        activations_encoder_h = get_activations(encoder, 1, reshaped_seq)

        neuron_output = activations_encoder_h[0]
        # print "neuron output", neuron_output
        seq_hidden_val = np.append(sequential_input, neuron_output)
        #print "Output", seq_hidden_val
        np.savetxt(f, seq_hidden_val.reshape(1,step_size+1), delimiter=',')

    f.close()


    f = file('networkTestData.csv', 'a')
    # Now take the rest of 50 flights for the test data

    for i in range(0,flight_test_size) :
        sequential_input = x_test[i,:]
        #do prediction on this
        #print "sequence grabbed", sequential_input
        reshaped_seq = np.reshape(sequential_input, (1,step_size))
        encoded_flight = encoder.predict(reshaped_seq)
        decoded_flight = decoder.predict(encoded_flight)

        activations_encoder_h = get_activations(encoder, 1, reshaped_seq)

        neuron_output = activations_encoder_h[0]
        # print "neuron output", neuron_output
        seq_hidden_val = np.append(sequential_input, neuron_output)
        #print "Output", seq_hidden_val
        np.savetxt(f, seq_hidden_val.reshape(1,step_size+1), delimiter=',')

    f.close()


def get_activations(model, layer, X_batch):

    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])

    return activations

if __name__ == "__main__":
    epochs = 200
    x_train, x_test = load_data()
    autoencoder, encoder, decoder = buildModel()
    autoencoder = train(autoencoder,  epochs)

    predict(autoencoder, encoder, decoder, x_train, x_test)
