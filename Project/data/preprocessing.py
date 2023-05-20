import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    features = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_11',\
        'sensor_12', 'sensor_13', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
    training_data = pd.read_csv('./features/training_normalized.csv')
    testing_data = pd.read_csv('./features/testing_normalized.csv')
    target_values = pd.read_csv('./data/RUL_FD001.txt', header=None)

    return training_data, testing_data, target_values, features


def plot_signal(df: pd.DataFrame, signal: str, 
                debugging: bool = False, test_set: bool = False):
    sensor_dictionary = {'sensor_1': '(Fan inlet temperature) ($^◦$R)',
        'sensor_2': '(LPC outlet temperature) ($^◦$R)',
        'sensor_3': '(HPC outlet temperature) ($^◦$R)',
        'sensor_4': '(LPT outlet temperature) ($^◦$R)',
        'sensor_5': '(Fan inlet Pressure) (psia)',
        'sensor_6': '(bypass-duct pressure) (psia)',
        'sensor_7': '(HPC outlet pressure) (psia)',
        'sensor_8': '(Physical fan speed) (rpm)',
        'sensor_9': '(Physical core speed) (rpm)',
        'sensor_10': '(Engine pressure ratio(P50/P2)',
        'sensor_11': '(HPC outlet Static pressure) (psia)',
        'sensor_12': '(Ratio of fuel flow to Ps30) (pps/psia)',
        'sensor_13': '(Corrected fan speed) (rpm)',
        'sensor_14': '(Corrected core speed) (rpm)',
        'sensor_15': '(Bypass Ratio) ',
        'sensor_16': '(Burner fuel-air ratio)',
        'sensor_17': '(Bleed Enthalpy)',
        'sensor_18': '(Required fan speed)',
        'sensor_19': '(Required fan conversion speed)',
        'sensor_20': '(High-pressure turbines Cool air flow)',
        'sensor_21': '(Low-pressure turbines Cool air flow)'}
    
    x_axis = 'time' if test_set is True else 'RUL'

    plt.figure(figsize=(13,5))
    for i in df['unit_number'].unique():
        if (not debugging) or (i % 13 == 0):   #For a better visualisation, we plot the sensors signals of 20 units only
            plt.plot(x_axis, (signal), data=df[df['unit_number']==i].rolling(10).mean())

    if test_set is False:
        plt.xlim(250, 0)  # reverse the x-axis so RUL counts down to zero
    plt.xticks(np.arange(0, 300, 25))
    try:
        plt.ylabel(signal+': '+sensor_dictionary[signal])
    except:
        plt.ylabel(signal)
    plt.xlabel(xlabel=x_axis)
    plt.show()
        

def preprocessing(debugging: bool = False, plotting: bool = False):
    unit = ['unit_number','time']
    settings = ['operational_setting 1','operational_setting 2','operational_setting 3']
    sensors = [f"sensor_{i}" for i in range(1, 22)]

    columns = unit+settings+sensors
    pwd = './data/'

    # Load data for FD001
    training_data = pd.read_csv(pwd+'train_FD001.txt', delimiter=' ', header=None, index_col=False, names=columns)
    testing_data = pd.read_csv(pwd+'test_FD001.txt', delimiter=' ', header=None, index_col=False, names=columns)
    target_values = pd.read_csv(pwd+'RUL_FD001.txt', header=None)

    def add_RUL_column(df):
        train_grouped_by_unit = df.groupby(by=unit[0]) 
        max_time_cycles = train_grouped_by_unit['time'].max() 
        merged = df.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on=unit[0],right_index=True)
        merged["RUL"] = merged["max_time_cycle"] - merged['time']
        merged = merged.drop("max_time_cycle", axis=1) 
        return merged

    # add RUL to training data
    training_data = add_RUL_column(training_data)

    # useful features from selection
    desired_features = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_11',\
            'sensor_12', 'sensor_13', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
    

    # drop features:
    for column in training_data.columns[2:-1]:
        if column not in desired_features:
            training_data = training_data.drop(columns=column)
    for column in testing_data.columns[2:]:
        if column not in desired_features:
            testing_data = testing_data.drop(columns=column)
            
    
    def GaussianKernel(x: float, sigma: float):
        kernel = np.exp(-(x**2)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
        return kernel

    def Regression(x, x_sample, y_sample, alpha, k):
        y = np.zeros_like(x)
        # x_sample_norm = (x_sample - x_sample.mean()) / x_sample.std()
        x_norm = x # (x - x_sample.mean()) / x_sample.std()

        for idx, y_i in enumerate(y):
            kernel_sum = 0
            for x_sample_i, y_sample_i in zip(x_sample, y_sample):
                y_i += y_sample_i*GaussianKernel(x_norm[idx]-x_sample_i, alpha)
                kernel_sum += GaussianKernel(x_norm[idx]-x_sample_i, alpha)
            y[idx] = y_i / kernel_sum
        return y

    df = training_data
    print("Filtering Training Set")
    for sensor in desired_features:
        print("    Filtering ",sensor)
        for i in df['unit_number'].unique():
            x_sample = df[df['unit_number']==i]['RUL'].to_numpy()
            y_sample = df[df['unit_number']==i][sensor].to_numpy()
            x = np.linspace(x_sample.max(), x_sample.min(), num=np.size(x_sample))

            for sigma in [20]:
                y = Regression(x,x_sample,y_sample, sigma, i)
                df.loc[df['unit_number']==i, sensor] = y  
    training_data_smooth = df

    dft = testing_data
    print("Filtering Test Set")
    for sensor in desired_features:
        print("    Filtering ",sensor)
        for i in dft['unit_number'].unique():
            x_sample = dft[dft['unit_number']==i]['time'].to_numpy()
            y_sample = dft[dft['unit_number']==i][sensor].to_numpy()
            x = np.linspace(x_sample.min(), x_sample.max(), num=np.size(x_sample))

            for sigma in [20]:
                y = Regression(x,x_sample,y_sample, sigma, i)
                dft.loc[dft['unit_number']==i, sensor] = y  

    testing_data_smooth = dft
    return training_data_smooth, testing_data_smooth, target_values, desired_features


def normalize(training_data, testing_data, _, features):
    train_normalized = pd.DataFrame()
    test_normalized = pd.DataFrame()

    train_normalized = training_data.copy()
    test_normalized = testing_data.copy()

    for sensor in features:
        train_normalized[sensor] = (train_normalized[sensor] - training_data[sensor].mean()) / training_data[sensor].std()
        test_normalized[sensor] = (test_normalized[sensor] - training_data[sensor].mean()) / training_data[sensor].std()

    return train_normalized, test_normalized


def main():
    done_smoothing = True

    if not done_smoothing:
        training_data, testing_data, target_values, features = preprocessing(plotting=True)

        for feature in features:
            plot_signal(training_data, feature, test_set=False)
            plot_signal(testing_data, feature, test_set=True)
        
        training_data.to_csv('./features/training_smoothed.csv')
        testing_data.to_csv('./features/testing_smoothed.csv')
    else:
        features = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_11',\
            'sensor_12', 'sensor_13', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
        training_data = pd.read_csv('./features/training_smoothed.csv')
        testing_data = pd.read_csv('./features/testing_smoothed.csv')
        target_values = pd.read_csv('./data/RUL_FD001.txt', header=None)

    training_data_normalized, testing_data_normalized = normalize(training_data, testing_data, target_values, features)

    training_data_normalized.to_csv('./features/training_normalized.csv')
    testing_data_normalized.to_csv('./features/testing_normalized.csv')

    for feature in features:
        plot_signal(training_data_normalized, feature, test_set=False)
        plot_signal(testing_data_normalized, feature, test_set=True)


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    exit(main())