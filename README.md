# CPSC 479 PROJECT 2 
Erick Juarez juarez.erick16@csu.fullerton.edu
## Problem 
This program was inspired by my passion for long distance running. I've been recording most of my runs with a GPS watch for the past 4 years, and I gathered data for over 1600 runs. The problem that I will be trying to solve is classifying these runs based on the run type. Training is different every day, but runnig produces very similar data for all run types. For simplicity, I decided to only classify 3 types of runs: Distance runs, Long Runs, and Workots. I decided to use a K-Nearest Neighbor algorithm because of its simplicity. I also decided to use mpi4py as the MPI standard for multicore processing. 

![alt text](https://github.com/EvilErick16/knn-running/blob/master/classes.JPG)

In the actual program, classes are as follows: Distance runs = 0, Long Runs = 1, Workouts = 2
## Pseudocode 
- Import libraries and declare shared variables
- Master Process: 
  - Import *training_data* and *test_point* from csv file
  - Partition *training_data* into P equal size segments. P being the number of processors used to run the program
  - call scatter() to send partitions of *training_data* to each process in the communicator 
  - call broadcast() to send *test_point* to all processes in the communicator 
 - Every process computes Eucledian distance between *test_point* and each point in their partition of *training_data*
 - Master process calls gather() to collect distances from other processes 
 - Master process selects **k** points from *training_data* with the shortest distance to *test_point*
 - Master process assigns *test_point* to the class where most of *test_point's* **K** neighbors belong to 

## Running the program 
### Requirements
This project was developed on Linux (Ubuntu 18.04LTS). with the following tools: 
1. Python 3.6.9
2. mpi4py 3.0.3 
3. pandas 1.0.3
4. numpy 1.18.4

If you don't want to install a lot of packages on your default system, I'd recommend having python installed and using a virtual envrionment. To create a virtual envrionment, run the followig command:

```pyhon3 -m venv path/to/location/<env-name>```

To activate your new envrionment run the following on a bash terminal: 

```source path/to/location/<env-name>/bin/activate```

you will now see `(env-name)` before the command prompt in your terminal. Install the required packages in this environment by running: 

```pip install mpi4py pandas numpy```

### Generating test file and runnig classifier  
Before running the actual classifier, you might want to get a datapoint to be the *test_point*. `generate_test.py` gets a random datapoint from the training data set, `training_set.csv`, and saves it in `testing_set.csv`. The main program, `knn-classifier.py`, uses `training_set.csv` to classify the generated `testing_set.csv`. 

Generate a test datapoint with the following command:

```python generate_test.py```

Then run the classifier: 

```mpirun -n 4 python knn-classifier.py```

replace *4* with the number of processors you wish to run the program with 
