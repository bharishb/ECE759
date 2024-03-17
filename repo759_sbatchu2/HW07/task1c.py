import matplotlib.pyplot as plt
import numpy as np

y = []
y2 = []
y3 = []
x = []

with open('task1c_thrust.log', 'r') as file:
    for line in file:
        y.append(float(line))
with open('task1c_cub.log', 'r') as file:
    for line in file:
        y2.append(float(line))
with open('task1c_naive_cuda.log', 'r') as file:
    for line in file:
        y3.append(float(line))


for i in range(10,21): #21 not inclusive
    x.append(f"{i}")

plt.figure(1)
plt.plot(x,np.log(y))
plt.xlabel('n = 2**i - log axis')
plt.ylabel('Time Spent(ms) - log axis')
plt.title('reduce - thrust API')
plt.savefig('task1c_thrust.png')

plt.figure(2)
plt.plot(x,np.log(y2))
plt.xlabel('n = 2**i - log axis')
plt.ylabel('Time Spent(ms) - log axis')
plt.title('reduce - cub')
plt.savefig('task1c_cub.png')

plt.figure(3)
plt.plot(x,np.log(y3))
plt.xlabel('n = 2**i - log axis')
plt.ylabel('Time Spent(ms) - log axis')
plt.title('reduce - naive cuda')
plt.savefig('task1c_naive_cuda.png')
#
plt.figure(4)
plt.xlabel('n = 2**i - log axis')
plt.ylabel('Time Spent(ms) - log axis')
plt.plot(x,np.log(y), label='Thrust')
plt.plot(x,np.log(y2), label='CUB')
plt.plot(x,np.log(y3), label='Naive cuda')
plt.title('Reduce scaling analysis across Implementations')
plt.legend()
plt.savefig('task1c_reduce_compare.png')
plt.show()
