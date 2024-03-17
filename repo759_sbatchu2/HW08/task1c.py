import matplotlib.pyplot as plt
import numpy as np

y = []
x = []

with open('task1c_time.log', 'r') as file:
    for line in file:
        y.append(float(line))


for i in range(1,21): #21 not inclusive
    x.append(f"{i}")
plt.figure(1)
plt.plot(x,y)
plt.xlabel('n = 1024, number of threads - t')
plt.ylabel('Time Spent(ms)')
plt.title('mmul - Matrix multiplication')
plt.savefig('task1c.png')
plt.show()
