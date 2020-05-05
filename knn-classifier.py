import pandas
from mpi4py import MPI


df = pd.read_csv("activities.csv")
print(df[['MovingTime', 'Distance', 'MaxSpeed', 'ElevationGain', 'Class']])