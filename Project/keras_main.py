import numpy as np
import pandas as pd

from keras.layers import Input, Dense
from keras.optimizers import rmsprop_v2
from keras.models import Model, load_model

from data.preprocessing import load_data
from data.preprocessing import preprocessing
from data.preprocessing import normalize

def OurModel(input_shape, output_shape):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    
    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with 1 node (RUL)
    X = Dense(output_shape, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='DQN_model')
    model.compile(loss="mse", optimizer=rmsprop_v2.RMSProp(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model

class NNAgent:
    def __init__(self, training_data, testing_data, target_values, features, epochs: int = 1000, learning_rate = 0.1):
        self.training_data = training_data
        self.testing_data = testing_data
        self.target_values = target_values
        self.features = features
        self.no_epochs = epochs

        # create main model
        self.model = OurModel(input_shape=(2*len(features),), output_shape=1)

    def train(self):
        mean_sample_time = self.testing_data.groupby('unit_number')['time'].max().mean()
        stdv_sample_time = self.testing_data.groupby('unit_number')['time'].max().std()

        n = int(mean_sample_time**2 / (mean_sample_time - stdv_sample_time))
        p = (1-stdv_sample_time/mean_sample_time)

        for epoch in range(self.no_epochs):
            train_data = np.empty((101,2*len(self.features)))
            target = np.empty((101,1))
            for engine in range(1,101):
                T = np.random.binomial(n,p)

                # Sample from trainingdata here!!!!

                

        ## PUT FEATURES HERE!!!

                feature_vector = np.zeros((1,2*len(self.features)))
                target = np.zeros((1,1))
                self.model.fit(feature_vector, target, batch_size=1, verbose=1)
    
    # check keras for engineers


def main() -> None:
    load_preprocessed_data: bool = True

    if load_preprocessed_data is not True:
        # Perform preprocessing and normalization
        training_data, testing_data, target_values, features = preprocessing(plotting=True)
        training_data, testing_data = normalize(training_data, testing_data, target_values, features)
    else:
        training_data, testing_data, target_values, features = load_data()
    
    agent = NNAgent(training_data, testing_data, target_values, features)
    agent.train()


    return


if __name__ == '__main__':
    exit(main())