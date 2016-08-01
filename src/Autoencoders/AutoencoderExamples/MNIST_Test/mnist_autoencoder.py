from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


def load_MNIST():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print x_train.shape
    print x_test.shape

    return x_train, x_test

def buildModel():

    # Size of the encoded representations
    # i.e. number of hidden units
    encoding_dim = 32
    input_dim = 784

    # ---------------------------------------------------------
    # Construct encoder and decoder layers
    # ---------------------------------------------------------
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
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

def train(autoencoder, encoder, decoder, epochs):

    autoencoder.fit(x_train, x_train,
                    nb_epoch=epochs,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # enode and decode some digits
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    return encoded_imgs, decoded_imgs


def plotMNIST(x_test, decoded_imgs):

    n = 10  # Number of digits to display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # original mnist data
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
   epochs = 1
   x_train, x_test = load_MNIST()
   autoencoder, encoder, decoder = buildModel() 
   encoded_imgs, decoded_imgs = train(autoencoder, encoder, decoder, epochs)
   plotMNIST(x_test, decoded_imgs)