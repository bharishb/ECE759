import matplotlib.pyplot as plt
import numpy as np

y = []
x = []

with open('task2c.log', 'r') as file:
    for line in file:
        y.append(float(line))


for i in range(5,21): #21 not inclusive
    x.append(f"{i}")

plt.figure(1)
plt.plot(x,np.log(y))
plt.xlabel('n = 2**i - log axis')
plt.ylabel('Time Spent(ms) - log axis')
plt.title('Count Unique elements : scaling analysis')
plt.savefig('task2c.png')
plt.show()
