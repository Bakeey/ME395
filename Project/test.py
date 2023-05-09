import pandas as pd
import numpy as np

unit = ['unit number','time']
settings = ['operational setting 1','operational setting 2','operational setting 3']
sensors = [f"sensor {i}" for i in range(1, 22)]

columns = unit+settings+sensors

# Load data for FD001
train_FD001 = pd.read_csv('train_FD001.txt', delimiter=' ', header=None, index_col=False, names=columns)
test_FD001 = pd.read_csv('test_FD001.txt', delimiter=' ', header=None, index_col=False, names=columns)
RUL_FD001 = pd.read_csv('RUL_FD001.txt', header=None)


# Load data for FD002
train_FD002 = pd.read_csv('train_FD002.txt', delimiter=' ', header=None, index_col=False, names=columns)
test_FD002 = pd.read_csv('test_FD002.txt', delimiter=' ', header=None, index_col=False, names=columns)
RUL_FD002 = pd.read_csv('RUL_FD002.txt', header=None)

# Load data for FD003
train_FD003 = pd.read_csv('train_FD003.txt', delimiter=' ', header=None, index_col=False, names=columns)
test_FD003 = pd.read_csv('test_FD003.txt', delimiter=' ', header=None, index_col=False, names=columns)
RUL_FD003 = pd.read_csv('RUL_FD003.txt', header=None)

# Load data for FD004
train_FD004 = pd.read_csv('train_FD004.txt', delimiter=' ', header=None, index_col=False, names=columns)
test_FD004 = pd.read_csv('test_FD004.txt', delimiter=' ', header=None, index_col=False, names=columns)
RUL_FD004 = pd.read_csv('RUL_FD004.txt', header=None)

# Add unit names to RUL
for RUL in [RUL_FD001, RUL_FD002, RUL_FD003, RUL_FD004]:
    RUL.insert(0, '-', [unit_name for unit_name in range(1,RUL.shape[0]+1)], True)
    RUL.columns = [unit[0],'RUL']


print(1)