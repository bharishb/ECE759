import matplotlib.pyplot as plt
import numpy as np

y1 = []
y2 = []
x = []

with open('lstm_cpp_scaling_time.log', 'r') as file:
    for line in file:
        y1.append(float(line))
with open('lstm_openmp_scaling_time.log', 'r') as file:
    for line in file:
        y2.append(float(line))


for i in range(10,21): #21 not inclusive
    x.append(f"2**{i}")

plt.figure(1)
plt.plot(x,np.log(y1), label = "cpu cpp")
plt.plot(x,np.log(y2), label = 'openmp')
plt.xlabel('t = 10 threads. Batch size : 1000, num_inputs -> increases 2**i')
plt.ylabel('Time Spent(ms)')
plt.legend()
plt.title('openmp vs cpu : log log plot')
plt.savefig('openmp_cpu_log.png')
plt.figure(2)
plt.plot(x,(y1), label = "cpu cpp")
plt.plot(x,(y2), label = 'openmp')
plt.xlabel('t = 10 threads. Batch size : 1000, num_inputs -> increases 2**i')
plt.ylabel('Time Spent(ms)')
plt.legend()
plt.title('openmp vs cpu')
plt.savefig('openmp_cpu.png')


plt.show()
