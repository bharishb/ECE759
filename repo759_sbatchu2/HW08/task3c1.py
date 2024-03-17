import matplotlib.pyplot as plt
import numpy as np

y = []
x = []

with open('task3c1_time.log', 'r') as file:
    for line in file:
        y.append(float(line))


for i in range(1,11): #11 not inclusive
    x.append(f"{i}")

plt.figure(1)
plt.plot(x,y)
plt.xlabel('n = 10^6, number of threads t = 8, ts = 2^{i}, i : 1 to 10 - log scale')
plt.ylabel('Time Spent(ms)')
plt.title('msort : Time vs threshold')
plt.savefig('task3c1.png')
plt.show()
