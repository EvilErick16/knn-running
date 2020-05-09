# Erick Juarez CPSC 479 SEC 1
# This is the main python file containing the K-nearest neighbor classifier

# Import libraries 
import numpy as np
import pandas as pd
from mpi4py import MPI      # MPI_Init() automatically called when you import MPI from library
from math import sqrt, ceil

# Distance function returns Euclidean distance between two datapoints with same number of features 
def getDistance(x, y):
    distance = 0
    for i in range(len(x) - 1):
        distance += ((x[i] - y[i]) ** 2)
    return sqrt(distance)

# Initialize multiprocess communication, each process is identified by it's rank 
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize global vairables 
training_set = []
partitions = []
testing_set = []
distances = []
k = 9

# Tasks for the master rank
if rank == 0: 
    # Import training and testing data 
    df = pd.read_csv("training_set.csv")
    training_set = df[['MovingTime', 'Distance', 'MaxSpeed', 'ElevationGain', 'Class']].to_numpy()
    df = pd.read_csv("testing_set.csv")
    testing_set = df[['MovingTime', 'Distance', 'MaxSpeed', 'ElevationGain', 'Class']].to_numpy()

    # Partition training data to send to worker processes 
    pts_per_proc = len(training_set)//size

    for i in range(size):
        partitions.append([])
    for i in range(len(training_set)):
        if i < pts_per_proc: 
            partitions[0].append(training_set[i])
        elif i < (pts_per_proc * 2):
            partitions[1].append(training_set[i])
        elif i < (pts_per_proc * 3):
            partitions[2].append(training_set[i])
        else:
            partitions[3].append(training_set[i])
    
# Scatter training data to worker porcesses 
partitions = comm.scatter(partitions, root=0)
# Broadcast testing data to worker processes 
testing_set = comm.bcast(testing_set, root=0)

# Each process calculataes distance from testing data to training data 
for point in partitions:
    distances.append(getDistance(point, testing_set[0]))
#print(rank, "-", partitions[0])

# Gather results from worker processes 
distances = comm.gather(distances, root=0)


# Master process does the rest of the work
if rank == 0:
    # Choose k closest points to training point
    closest = []    # array of closest indices 
    k_dist = []     # distance of the k closest neigbors 
    dist_vect = []  # distances from training set and testing point

    # Reshape data gathered from other processors
    for i in range(len(distances)):
        for j in range(len(distances[i])):
            dist_vect.append(distances[i][j])

    # create a copy of distance array and sort it to get closest values
    k_dist = dist_vect.copy()
    k_dist = np.sort(k_dist)

    # Find the corresponding datapoints from closest values 
    for i in range(k): 
        for j in range(len(dist_vect)):
            if k_dist[i] == dist_vect[j]:
                closest.append(j)
   
    # Classify test point  
    classes = [0, 0, 0]
    for i in closest: 
        if training_set[i][4] == 0: 
            classes[0] += 1
        elif training_set[i][4] == 1: 
            classes[1] += 1
        elif training_set[i][4] == 2: 
            classes[2] += 1
    predicted = classes.index(max(classes))
    actual = int(testing_set[0][4])
    print("--------------------------------")
    print("Training points -", len(dist_vect))
    print("              K -", k)
    print("      Predicted -", predicted)
    print("         Actual -", actual)
    print("--------------------------------")
 