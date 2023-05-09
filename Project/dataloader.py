import pandas as pd
import numpy as np

def load_data(pwd) -> tuple[pd.DataFrame]:
    dfs = []
    unit = ['unit_number','time']
    settings = ['operational_setting 1','operational_setting 2','operational_setting 3']
    sensors = [f"sensor_{i}" for i in range(1, 22)]

    columns = unit+settings+sensors

    # Load data for FD001
    train_FD001 = pd.read_csv(pwd+'train_FD001.txt', delimiter=' ', header=None, index_col=False, names=columns)
    test_FD001 = pd.read_csv(pwd+'test_FD001.txt', delimiter=' ', header=None, index_col=False, names=columns)
    RUL_FD001 = pd.read_csv(pwd+'RUL_FD001.txt', header=None)

    # Load data for FD002
    train_FD002 = pd.read_csv(pwd+'train_FD002.txt', delimiter=' ', header=None, index_col=False, names=columns)
    test_FD002 = pd.read_csv(pwd+'test_FD002.txt', delimiter=' ', header=None, index_col=False, names=columns)
    RUL_FD002 = pd.read_csv(pwd+'RUL_FD002.txt', header=None)

    # Load data for FD003
    train_FD003 = pd.read_csv(pwd+'train_FD003.txt', delimiter=' ', header=None, index_col=False, names=columns)
    test_FD003 = pd.read_csv(pwd+'test_FD003.txt', delimiter=' ', header=None, index_col=False, names=columns)
    RUL_FD003 = pd.read_csv(pwd+'RUL_FD003.txt', header=None)

    # Load data for FD004
    train_FD004 = pd.read_csv(pwd+'train_FD004.txt', delimiter=' ', header=None, index_col=False, names=columns)
    test_FD004 = pd.read_csv(pwd+'test_FD004.txt', delimiter=' ', header=None, index_col=False, names=columns)
    RUL_FD004 = pd.read_csv(pwd+'RUL_FD004.txt', header=None)


    # Add unit names to RUL
    for RUL in [RUL_FD001, RUL_FD002, RUL_FD003, RUL_FD004]:
        RUL.insert(0, '-', [unit_name for unit_name in range(1,RUL.shape[0]+1)], True)
        RUL.columns = [unit[0],'RUL']
        
    return (train_FD001, test_FD001, RUL_FD001)

def main():
    data_location = './data/'
    train, test, target = load_data(data_location)

    ## TODO: For Training data -- Plot all sensor data, plot distributions of RUL, plot correlations, see Pandas example

    return

if __name__=='__main__':
    exit(main())