import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import Input, Dense
from keras.optimizers import rmsprop_v2
from keras.models import Model, load_model
from keras import regularizers

from data.preprocessing import load_data
from data.preprocessing import preprocessing
from data.preprocessing import normalize

def OurModel(input_shape, output_shape):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(1024, input_shape=input_shape, activation="linear",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5))(X_input)

    X = Dense(1024, activation="relu",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5))(X)

    X = Dense(1024, activation="relu",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5))(X)

    X = Dense(512, activation="relu",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5))(X)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5))(X)
    
    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5))(X)

    # Output Layer with 1 node (RUL)
    X = Dense(output_shape, activation="relu",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5))(X)

    model = Model(inputs = X_input, outputs = X, name='DQN_model')
    model.compile(loss="mse", optimizer=rmsprop_v2.RMSProp(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["mean_absolute_error"])

    model.summary()
    return model

class NNAgent:
    def __init__(self, training_data, testing_data, target_values, features, epochs: int = 100, training_size: int = 1000, learning_rate: float = 0.1):
        self.training_data = training_data
        self.testing_data = testing_data
        self.target_values = target_values
        self.features = features
        self.no_epochs = epochs
        self.training_size = training_size

        # create main model
        self.model = OurModel(input_shape=(2*len(features),), output_shape=1)

    def generate_features(self, dataset, training_size: int = None, testing: bool = False):
        if training_size is None:
            training_size = self.training_size
        
        if not testing:
            mean_sample_time = self.target_values.mean() # self.testing_data.groupby('unit_number')['time'].max().mean()
            stdv_sample_time = self.target_values.std() # self.testing_data.groupby('unit_number')['time'].max().std()

            max_sample_time = self.target_values.max() # self.testing_data.groupby('unit_number')['time'].max().mean()
            min_sample_time = self.target_values.min() # self.testing_data.groupby('unit_number')['time'].max().std()

            mean_start_time = self.testing_data.groupby('unit_number')['time'].max().mean()
            stdv_start_time = self.testing_data.groupby('unit_number')['time'].max().std()

            n = int(mean_sample_time**2 / (mean_sample_time - stdv_sample_time))
            p = float(1-stdv_sample_time/mean_sample_time)

            n_2 = int(mean_start_time**2 / (mean_start_time - stdv_start_time))
            p_2 = float(1-stdv_start_time/mean_start_time)

            feature_vector = np.empty((training_size,2*len(self.features)))
            engine_numbers = np.random.randint(1,self.training_data['unit_number'].max() + 1, (training_size,))

            target_vector = np.empty((training_size,1))
        
        else:
            engine_numbers = np.arange(1,self.testing_data['unit_number'].max() + 1)
            feature_vector = np.empty((self.testing_data['unit_number'].max(),2*len(self.features)))
            target_vector = None

        for idx,engine in enumerate(engine_numbers):
            if not testing:
                timeseries_length = self.training_data.loc[self.training_data['unit_number']==engine]['time'].max()
                T_end = int(np.clip(timeseries_length - np.random.randint(min_sample_time, max_sample_time), 1, \
                        timeseries_length))
                
                T_start = np.clip(T_end - np.random.binomial(n_2,p_2), 1, \
                        T_end)
                
                mean_start_time = self.testing_data.groupby('unit_number')['time'].max().mean()
                stdv_start_time = self.testing_data.groupby('unit_number')['time'].max().std()
                
                target_vector[idx] = self.training_data.loc[ 
                    (self.training_data['unit_number']==engine) &\
                    (self.training_data['time']==T_end), 'RUL'].to_numpy()
                # T = np.random.randint(1, \
                #         self.training_data.loc[self.training_data['unit_number']==engine]['time'].max())
            else:
                T_end = self.testing_data.loc[(self.testing_data['unit_number']==engine), 'time'].max()
                T_start = 1

            feature_vector[idx, :len(self.features)] = dataset.loc[ 
                (dataset['unit_number']==engine) & (dataset['time']==T_end), self.features].to_numpy()
            
            feature_vector[idx, len(self.features):2*len(self.features)] = dataset.loc[ 
                (dataset['unit_number']==engine) & (dataset['time']==T_start), self.features].to_numpy()
            
            # feature_vector[idx, 2*len(self.features):3*len(self.features)] = dataset.loc[ 
            #     dataset['unit_number']==engine, self.features].mean().to_numpy()
            # 
            # feature_vector[idx, 3*len(self.features):4*len(self.features)] = dataset.loc[ 
            #     dataset['unit_number']==engine, self.features].median().to_numpy()
            # 
        return feature_vector, target_vector
        
             
                
    def train(self):
        train_data, target = self.generate_features(self.training_data)
        history = self.model.fit(train_data, target, validation_split = 0.25, batch_size=25, epochs = self.no_epochs, verbose=1)

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        abs_error = history.history['mean_absolute_error']
        val_abs_error = history.history['val_mean_absolute_error']

        return loss, val_loss, abs_error, val_abs_error


    def predict(self): # TODO predit better? How many epochs?
        # TODO 
        # MAKE TRAINING DATA MORE LIKE TESTING DATA, BUT HOWW????
        feature_vector, _ = self.generate_features(self.testing_data, testing=True)
            
        prediction = self.model.predict(feature_vector)

        mean_abs_error = np.mean(np.abs(prediction - self.target_values))

        target = self.target_values.to_numpy().reshape(100,)
        prediction = prediction.reshape(100,)
        indexes = np.argsort(target)
        prediction = [prediction[ii] for ii in indexes]
        target = [target[ii] for ii in indexes]


        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(100)
        bar_width = 0.4
        b1 = ax.bar(x + bar_width,target,
            width=bar_width, label = 'target value')
        b2 = ax.bar(x, prediction,
                    width=bar_width, label = 'prediction')
        plt.legend()


        # Fix the x-axes.
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(np.arange(1,101))
        ax.set_title(r'no_epochs = '+str(self.no_epochs)+r', training data size = '+str(self.training_size))
        # fig.show()

            
        return prediction, mean_abs_error


def main() -> None:
    load_preprocessed_data: bool = True

    if load_preprocessed_data is not True:
        # Perform preprocessing and normalization
        training_data, testing_data, target_values, features = preprocessing(plotting=True)
        training_data, testing_data = normalize(training_data, testing_data, target_values, features)
    else:
        training_data, testing_data, target_values, features = load_data()
    
    epochs = [100] # [10,20,50,100,200,500]
    training_size = [500] #  [100,500,1000,5000]

    result = np.empty((len(epochs),len(training_size),))
    mean_abs_error = np.empty((len(epochs),len(training_size)))

    for ii,no_epoch in enumerate(epochs):
        for jj,train_size in enumerate(training_size):
            print(no_epoch, train_size, end='\n')
            started_training = False
            while (not started_training) or agent.predict()[1] > 75: # kill off dead gradients
                agent = NNAgent(training_data, testing_data, target_values, features, epochs=no_epoch, training_size=train_size)
                loss, val_loss, _, _ = agent.train()
                started_training = True
                _, mean_abs_error[ii,jj] = agent.predict()

    s = sns.heatmap(mean_abs_error, annot=True, xticklabels=training_size, yticklabels=epochs)
    s.set(xlabel='No. of Episodes', ylabel='Size of Training Data')
    plt.title("Mean Absolute Error of Testing Data")
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(loss)), loss, 'k', label='Training Loss')
    plt.plot(np.arange(len(loss)), val_loss, 'r', label='Validation Loss')
    plt.legend()
    plt.xlabel("No. Epochs")
    plt.ylim(0, max(loss))
    plt.show()



    return


if __name__ == '__main__':
    exit(main())