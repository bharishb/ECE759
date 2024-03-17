import matplotlib.pyplot as plt

y = []
y2 = []
x = []

with open('task1_time_t1024.log', 'r') as file:
    for line in file:
        y.append(float(line))
with open('task1_time_t32.log', 'r') as file:
    for line in file:
        y2.append(float(line))


for i in range(5,15): #30 not inclusive
    x.append(f"{i}")

plt.figure(1)
plt.plot(x,y)
plt.xlabel('2^n : size of the array. n shown on axis')
plt.ylabel('Time Spent(ms)')
plt.title('scaling analysis : THREADS PER BLOCK = 1024')
plt.savefig('task1_t1024.png')

plt.figure(2)
plt.plot(x,y2)
plt.xlabel('2^n : size of the array. n shown on axis')
plt.ylabel('Time Spent(ms)')
plt.title('scaling analysis : THREADS PER BLOCK = 32')
plt.savefig('task1_t32.png')
#
plt.figure(3)
plt.plot(x,y, label='THREADS PER BLOCK = 1024')
plt.plot(x,y2, label='THREADS PER BLOCK = 32')
plt.xlabel('2^n : size of the array. n shown on axis')
plt.ylabel('Time Spent(ms)')
plt.title('scaling analysis')
plt.legend()
plt.savefig('task1_t1024_t32.png')
plt.show()
