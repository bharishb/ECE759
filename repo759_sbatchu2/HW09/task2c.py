import matplotlib.pyplot as plt
import numpy as np

y = []
y2 = []
x = []

with open('task2c_time.log', 'r') as file:
    for line in file:
        y.append(float(line))
with open('task2c_time_simd.log', 'r') as file:
    for line in file:
        y2.append(float(line))

y = np.sum(np.reshape(y, (10,10)), axis=0)/10
y2 = np.sum(np.reshape(y2, (10,10)), axis=0)/10

for i in range(1,11): #11 not inclusive
    x.append(f"{i}")

plt.figure(1)
plt.plot(x,y)
plt.xlabel('n= 10**6, t : number of threads')
plt.ylabel('Time Spent(ms)')
plt.title('Montecarlo - without SIMD')
plt.savefig('task2c_without_simd.png')

plt.figure(2)
plt.plot(x,y2)
plt.xlabel('n= 10**6, t : number of threads')
plt.ylabel('Time Spent(ms)')
plt.title('Montecarlo - with SIMD')
plt.savefig('task2c_with_simd.png')

#
plt.figure(4)
plt.xlabel('n= 10**6, t : number of threads')
plt.ylabel('Time Spent(ms)')
plt.plot(x,y, label='without SIMD')
plt.plot(x,y2, label='with SIMD')
plt.title('Montecarlo - with & without SIMD')
plt.legend()
plt.savefig('task2c_simd_compare.png')
plt.show()
