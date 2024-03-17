import matplotlib.pyplot as plt
import numpy as np

y1 = []
y2 = []
y3 = []
y4 = []
x1 = []
x2 = []

with open('task2d1_time.log', 'r') as file:
    for line in file:
        y1.append(float(line))
with open('task2d1_time_pure_omp.log', 'r') as file:
    for line in file:
        y2.append(float(line))
with open('task2d2_time.log', 'r') as file:
    for line in file:
        y3.append(float(line))
with open('task2d2_time_pure_omp.log', 'r') as file:
    for line in file:
        y4.append(float(line))


for i in range(1,21): #21 not inclusive
    x1.append(f"{i}")
for i in range(1,27): #27 not inclusive
    x2.append(f"{i}")

plt.figure(1)
plt.plot(x1,y1, label = "openmp+mpi")
plt.plot(x1,y2, label = 'pure omp')
plt.xlabel('n= 10**7, t : number of threads')
plt.ylabel('Time Spent(ms)')
plt.legend()
plt.title('openmp + mpi vs pure omp')
plt.savefig('task2d1.png')

plt.figure(2)
plt.plot(x2,np.log(y3), label = "openmp+mpi")
plt.plot(x2,np.log(y4), label = 'pure omp')
plt.xlabel('t = 6, n = 2**1 to 2**26 (log scale)')
plt.ylabel('Time Spent(ms) - log scale')
plt.title('openmp+mpi vs Pure omp')
plt.legend()
plt.savefig('task2d2.png')
plt.show()
