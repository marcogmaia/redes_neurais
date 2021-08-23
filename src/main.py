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
test_CSV_path = './data/test.csv'
validation_CSV_path = './data/validation_data.csv'

numberOfColumns = 248

def getNumberOfColumnsIn(csvPath):
    df = pd.read_csv("./data/traning_data.csv")
    total_columns=len(df.axes[1]) #===> 1 for a column, 0 for row
    return total_columns

''' -- MAIN -- '''
# load the dataset
dataset = loadtxt('./data/traning_data.csv', delimiter=',')
# split into input (X) and output (y) variables
#[0:2] of [a, b, c] means [a, b].
#last two columns of input ignored, because they are the "answer" (?)
input = dataset[:,0:numberOfColumns-2]

#output is the last column
output = dataset[:,numberOfColumns-1] #last column

'''
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
'''
