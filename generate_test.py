# Erick Juarez CPSC 479 SEC 1
# This is a supplementary file used to randomly get a datapoint from training set to use as a training point 

# import libraries
import pandas as pd 
import random

# read data from training set
df = pd.read_csv("training_set.csv")

# create an empty dataframe 
data = {'ActivityDate':[], 
        'ActivityName':[],
        'ElapsedTime':[],
        'MovingTime':[],
        'Distance':[],
        'MaxSpeed':[],
        'ElevationGain':[],
        'Class':[]}
data = pd.DataFrame(data)

# choose a random row from the training set
rand = random.randint(0, len(df))
data = data.append(df[rand:rand+1], ignore_index=True)

# Create csv file 
data.to_csv("testing_set.csv")
print("---------------------------------------------")
print("Generated datapoint stored in testing_set.csv")
print("---------------------------------------------")
print(data[:])
print("---------------------------------------------")
