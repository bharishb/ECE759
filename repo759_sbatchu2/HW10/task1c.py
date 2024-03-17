import matplotlib.pyplot as plt
import numpy as np

y1 = []
y2 = []
y3 = []
y4 = []
x = []

with open('task1c_time_int_sum.log', 'r') as file:
    for line in file:
        y1.append(float(line))
with open('task1c_time_int_mult.log', 'r') as file:
    for line in file:
        y2.append(float(line))

with open('task1c_time_float_sum.log', 'r') as file:
    for line in file:
        y3.append(float(line))
with open('task1c_time_float_mult.log', 'r') as file:
    for line in file:
        y4.append(float(line))

for i in range(1,7): #7 not inclusive
    x.append(f"{i}")
plt.figure(1)
plt.plot(x,y1)
plt.xlabel('Reduce implementations 1 to 6')
plt.ylabel('Time Spent(ms)')
plt.title('Optimizations time taken - int sum')
plt.savefig('task1c_int_sum.png')
plt.figure(2)
plt.plot(x,y2)
plt.xlabel('Reduce implementations 1 to 6')
plt.ylabel('Time Spent(ms)')
plt.title('Optimizations time taken - int mult')
plt.savefig('task1c_int_mult.png')
plt.figure(3)
plt.plot(x,y3)
plt.xlabel('Reduce implementations 1 to 6')
plt.ylabel('Time Spent(ms)')
plt.title('Optimizations time taken - float sum')
plt.savefig('task1c_float_sum.png')
plt.figure(4)
plt.plot(x,y4)
plt.xlabel('Reduce implementations 1 to 6')
plt.ylabel('Time Spent(ms)')
plt.title('Optimizations time taken - float mult')
plt.savefig('task1c_float_mult.png')
plt.figure(5)
plt.plot(x,y1, label = 'int sum')
plt.plot(x,y2, label = 'int mult')
plt.plot(x,y3, label = 'float sum')
plt.plot(x,y4, label = 'float mult')
plt.xlabel('Reduce implementations 1 to 6')
plt.ylabel('Time Spent(ms)')
plt.legend()
plt.title('Optimizations time taken')
plt.savefig('task1c_all.png')
plt.show()



















