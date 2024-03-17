import matplotlib.pyplot as plt
import numpy as np

y = []
x = []

with open('task3b_time.log', 'r') as file:
    for line in file:
        y.append(float(line))


for i in range(1,26): #26 not inclusive
    x.append(f"{i}")
plt.figure(1)
plt.plot(x,np.log(y))
plt.xlabel('n = 2**{i}, i = 1 to 25 : log scale')
plt.ylabel('Time Spent(ms) - log scale')
plt.title('MPI Time taken vs n (message length)')
plt.savefig('task3b.png')
plt.show()
