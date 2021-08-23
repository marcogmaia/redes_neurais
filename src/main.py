'''
import pandas as pd
import zipfile

file_path = './data/data.zip'
zf =  zipfile.ZipFile(file_path, 'r')
fl = zf.filelist
for f in fl:
    print(f.filename)
'''


#Starting with MLP following this example:
#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
import pandas as pd
from numpy import loadtxt
#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Variables and Constants
training_CSV_path = './data/traning_data.csv' #should be 'training', not 'traning' :P
test_CSV_path = './data/test_data.csv'
validation_CSV_path = './data/validation_data.csv'

numberOfColumns = 248

isDebug = True

#Auxiliar functions
def getNumberOfColumnsIn(csvPath):
    df = pd.read_csv(csvPath)
    total_columns=len(df.axes[1]) #===> 1 for a column, 0 for row
    return total_columns

''' -- MAIN -- '''
# load the dataset skipping first row of table (headers)

max_rows = None
if isDebug:
    max_rows = 100

dataset = loadtxt(test_CSV_path, delimiter=',', skiprows=1, max_rows=max_rows)
# split into input (X) and output (y) variables
#[0:2] of [a, b, c] means [a, b].
#last two columns of input ignored, because they are the "answer" (?)
input = dataset[:,0:numberOfColumns-2]

#output is the last column
output = dataset[:,numberOfColumns-1] #last column

# define the keras model (Network structure)
model = Sequential()
model.add(Dense(units=12, input_dim=numberOfColumns-2, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
