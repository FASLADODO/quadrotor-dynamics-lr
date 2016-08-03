from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import os, sys



#(x_train, _), (x_test,_) = mnist.load_data()
#x_train = x_train.astype('float32')/255.
#x_test = x_test.astype('float32')/255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


path_to_dataset_X=os.path.abspath(os.path.join('../../..', 'data/NonlinearDynamicModel/speedDataNormalized.csv'))
speedDataNormalized = np.loadtxt(open(path_to_dataset_X,"rb"),delimiter=",");



x_train = speedDataNormalized

x_test = x_train # for now lets assume they are the same

print x_train.shape
print x_test.shape



## -----------------------------------------------------------

# Size of the encoded representation
encoding_dim = 50

# input place_holder
input_img = Input(shape=(4,))

# encoded is the encoded representation of the input
encoded = Dense(encoding_dim, activation='sigmoid')(input_img)

decoded = Dense(4, activation='linear')(encoded)

# This is the model that maps input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output = encoded)

# create a placehoder for an encoded (32-dimensional) input

encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]


# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss = 'mse')


autoencoder.fit(x_train, x_train, nb_epoch=1,
                batch_size=4,
                shuffle=True,
                validation_data=(x_test,x_test))

# enode and decode some digits
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


path_to_decoded_dataset =os.path.abspath(os.path.join('../../..', 'data/NonlinearDynamicModel/decodedSpeeds2.csv'))
np.savetxt('x_original.csv', x_test, delimiter=',')
np.savetxt(path_to_decoded_dataset, decoded_imgs, delimiter=',')


print "Original input data\n", x_test
print "Reconstructed output\n " , decoded_imgs


