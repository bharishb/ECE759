import matplotlib.pyplot as plt
import numpy as np

y = []
x = [512, 1024, 2048, 4096]

with open('task4b_time.log', 'r') as file:
    for line in file:
        y.append(float(line))


#for i in range(1,5): #5 not inclusive
#    x.append(f"{i}")
plt.figure(1)
plt.plot(x,y)
plt.xlabel('n value : linear scale')
plt.ylabel('Time Spent(ms)')
plt.title('convolve function vs n')
plt.savefig('task4b.png')
plt.show()
