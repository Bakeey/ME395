import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def return_filters(dt: float = 0.02) -> dict:
    """A dict of different filters"""
    filters = {
        'Euler_FW': np.array([0,-1,1]) / dt,
        'Euler_BW': np.array([-1,1,0]) / dt,
        'Central_Difference': np.array([-1,0,1]) / (2*dt),
        '2nd_Order': np.array([1,-2,1]) / (dt**2)
    }
    return filters


def convolution(input: np.ndarray, filter: np.ndarray, dt: float = 0.02) -> np.ndarray:
    """
    Performs convolution with a given filter on some input array
    """
    output = np.ones_like(input) # allocate storage
    input  = np.pad(input, (1,1), 'constant', constant_values=(0, 0)) # padding for edges
    
    # Perform Euler forward and backward at edges in any case
    edge_filter = return_filters(dt)
    output[0]  = np.dot(input[0:3], edge_filter['Euler_FW'])
    output[-1] = np.dot(input[-3:], edge_filter['Euler_BW'])

    # Convolution Operation
    for idx in range(1,len(output)-1):
        output[idx] = np.dot(filter,input[idx:idx+len(filter)])
    
    return output


def load_data(name: str = 'Chichi', path: str = "./src/SeismicWave.xlsx") -> tuple:
    """
    Loads data from XLSX to tuple of arrays
    """
    if name == 'Chichi':
        columns: str = "A:D"
        rows: int = 866
        df = pd.read_excel(path, header = 1, usecols=columns, nrows=rows)
        
        time            = df['t (s)'].to_numpy()
        displacement    = df['displacement (m)'].to_numpy()
        velocity        = df['velocity (m/s)'].to_numpy()
        acceleration    = df['acceleration (m/s2)'].to_numpy()

    elif name == 'Imperial':
        columns: str = "F:I"
        df = pd.read_excel(path, header = 1, usecols=columns)

        time            = df['t (s).1'].to_numpy()
        displacement    = df['displacement (m).1'].to_numpy()
        velocity        = df['velocity (m/s).1'].to_numpy()
        acceleration    = df['acceleration (m/s2).1'].to_numpy()

    return time, displacement, velocity, acceleration


def main() -> int:
    """
    Main Loop.
    """
    # Load Data, either 'Chichi' or 'Imperial'
    dataset: str = 'Imperial'
    time, displacement, velocity, acceleration = load_data(dataset)

    # Initial Plotting of Displacement Data
    plt.figure()
    plt.plot(time, displacement, color=[0.5,0.5,0.5])
    plt.legend(['Measurement Data'])
    plt.title(str(dataset)+" Earthquake Displacement Profile")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement (m)")
    plt.show()

    # Definition of filters
    euler_fw = return_filters()['Euler_FW']
    euler_bw = return_filters()['Euler_BW']
    central_diff = return_filters()['Central_Difference']
    second_order = return_filters()['2nd_Order']

    # Differentiation using convolution
    velocity_euler_fw = convolution(displacement,euler_fw)
    velocity_centrald = convolution(displacement,central_diff)

    # Plot Velocity Profile
    plt.figure()
    order = [0,1]
    if dataset == 'Chichi':
        order = [0,1,2]
        plt.plot(time, velocity, color=[0.5,0.5,0.5],
                 label='Measurement Data')
    else:
        velocity = np.zeros_like(velocity)
    plt.plot(time, velocity_centrald,'r--',
             label='From Displacement Profile (central diff.)')
    plt.plot(time, velocity_euler_fw,'b.', markersize=3,
             label='From Displacement Profile (forward diff.)')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.title(str(dataset)+" Earthquake Velocity Profile")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity (m/s)")
    plt.show()

    if dataset == 'Chichi':
        # Plot Velocity Error
        plt.figure()
        plt.plot(time, velocity_euler_fw - velocity,'b-',
                label='From Displacement Profile (forward diff.)')
        plt.plot(time, velocity_centrald - velocity,'r-',
                label='From Displacement Profile (central diff.)')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1,0]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
        plt.title(str(dataset)+" Earthquake Velocity Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity (m/s)")
        plt.show()

    # Differentiation to Acceleration from velocity OR displacement using convolution
    acceleration_euler_fw = convolution(velocity_euler_fw,euler_bw)
    acceleration_centrald = convolution(velocity_centrald,central_diff)
    acceleration_2nd_order= convolution(displacement,second_order)

    # Plot Acceleration
    plt.figure()
    order = [0,1,2]
    if dataset == 'Chichi':
        order = [0,1,2,3]
        plt.plot(time, acceleration, color=[0.5,0.5,0.5],
                 label='Measurement Data')
    plt.plot(time, acceleration_centrald,'r.',
             label='From Central Diff. Velocity Profile (central diff.)')
    plt.plot(time, acceleration_euler_fw,'b.', markersize=3,
             label='From Forward Diff. Velocity Profile (backward diff.)')
    plt.plot(time, acceleration_2nd_order,'y.', markersize=1,
             label='From Displacement Profile (2nd order central diff.)')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.title(str(dataset)+" Earthquake Acceleration Profile")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration (m/s2)")
    plt.show()

    # Plot Acceleration Error
    plt.figure()
    order = [0,1]
    plt.plot(time, acceleration_centrald - acceleration_2nd_order,'r-',
             label='From Central Diff. Velocity Profile (central diff.)')
    plt.plot(time, acceleration_euler_fw - acceleration_2nd_order,'b-', markersize=3,
             label='From Forward Diff. Velocity Profile (backward diff.)')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.title(str(dataset)+" Earthquake Acceleration Difference to Second-Order Central Difference")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration (m/s2)")
    plt.show()

    return 0

    
if __name__ == '__main__':
    main()
