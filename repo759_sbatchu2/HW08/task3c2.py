import matplotlib.pyplot as plt
import numpy as np

y = []
x = []

with open('task3c2_time.log', 'r') as file:
    for line in file:
        y.append(float(line))


for i in range(1,21): #21 not inclusive
    x.append(f"{i}")

plt.figure(1)
plt.plot(x,y)
plt.xlabel('n = 10^6, ts = 2^{10}, t : number of threads : 1 to 20')
plt.ylabel('Time Spent(ms)')
plt.title('msort : Time vs number of threads')
plt.savefig('task3c2.png')
plt.show()
