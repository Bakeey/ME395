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
    X = Dense(512, input_shape=input_shape, activation="linear")(X_input)

    X = Dense(512, activation="exponential")(X)

    X = Dense(512, activation="relu")(X)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu")(X)
    
    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu")(X)

    # Output Layer with 1 node (RUL)
    X = Dense(output_shape, activation="relu")(X)

    model = Model(inputs = X_input, outputs = X, name='DQN_model')
    model.compile(loss="mse", optimizer=rmsprop_v2.RMSProp(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["mean_absolute_error"])

    model.summary()
    return model

class NNAgent:
    def __init__(self, training_data, testing_data, target_values, features, epochs: int = 10, learning_rate = 0.1):
        self.training_data = training_data
        self.testing_data = testing_data
        self.target_values = target_values
        self.features = features
        self.no_epochs = epochs

        # create main model
        self.model = OurModel(input_shape=(4*len(features),), output_shape=1)

    def generate_features(self, dataset, training_size: int = 1000, testing: bool = False):
        if not testing:
            mean_sample_time = self.testing_data.groupby('unit_number')['time'].max().mean()
            stdv_sample_time = self.testing_data.groupby('unit_number')['time'].max().std()

            n = int(mean_sample_time**2 / (mean_sample_time - stdv_sample_time))
            p = (1-stdv_sample_time/mean_sample_time)

            feature_vector = np.empty((training_size,4*len(self.features)))
            engine_numbers = np.random.randint(1,self.training_data['unit_number'].max() + 1, (training_size,))

            target_vector = np.empty((training_size,1))
        
        else:
            engine_numbers = np.arange(1,self.testing_data['unit_number'].max() + 1)
            feature_vector = np.empty((self.testing_data['unit_number'].max(),4*len(self.features)))
            target_vector = None

        for idx,engine in enumerate(engine_numbers):
            if not testing:
                T = np.clip(np.random.binomial(n,p), 1, \
                        self.training_data.loc[self.training_data['unit_number']==engine]['time'].max())
                
                target_vector[idx] = self.training_data.loc[ 
                    (self.training_data['unit_number']==engine) &\
                    (self.training_data['time']==T), 'RUL'].to_numpy()
                # T = np.random.randint(1, \
                #         self.training_data.loc[self.training_data['unit_number']==engine]['time'].max())
            else:
                T = self.testing_data.loc[(self.testing_data['unit_number']==engine), 'time'].max()

            feature_vector[idx, :len(self.features)] = dataset.loc[ 
                (dataset['unit_number']==engine) & (dataset['time']==T), self.features].to_numpy()
            
            feature_vector[idx, len(self.features):2*len(self.features)] = dataset.loc[ 
                (dataset['unit_number']==engine) & (dataset['time']==1), self.features].to_numpy()
            
            feature_vector[idx, 2*len(self.features):3*len(self.features)] = dataset.loc[ 
                dataset['unit_number']==engine, self.features].mean().to_numpy()
            
            feature_vector[idx, 3*len(self.features):4*len(self.features)] = dataset.loc[ 
                dataset['unit_number']==engine, self.features].median().to_numpy()
            
        return feature_vector, target_vector
        
                
    def train(self):
        train_data, target = self.generate_features(self.training_data)
        self.model.fit(train_data, target, validation_split = 0.25, batch_size=250, epochs = 100, verbose=1)
        return


    def predict(self): # TODO predit better? How many epochs?
        feature_vector, _ = self.generate_features(self.testing_data, testing=True)
            
        prediction = self.model.predict(feature_vector)
            
        return prediction


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
    result = agent.predict()

    return


if __name__ == '__main__':
    exit(main())