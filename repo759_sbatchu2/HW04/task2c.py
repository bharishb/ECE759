import matplotlib.pyplot as plt

y = []
y2 = []
x = []

with open('task2_time_t1024.log', 'r') as file:
    for line in file:
        y.append(float(line))
with open('task2_time_t512.log', 'r') as file:
    for line in file:
        y2.append(float(line))


for i in range(10,30): 
    x.append(f"{i}")

plt.figure(1)
plt.plot(x,y)
plt.xlabel('2^n : size of the array. n shown on axis')
plt.ylabel('Time Spent(ms)')
plt.title('scaling analysis : THREADS PER BLOCK = 1024')
plt.savefig('task2_t1024.png')

plt.figure(2)
plt.plot(x,y2)
plt.xlabel('2^n : size of the array. n shown on axis')
plt.ylabel('Time Spent(ms)')
plt.title('scaling analysis : THREADS PER BLOCK = 512')
plt.savefig('task2_t512.png')
#
plt.figure(3)
plt.plot(x,y, label='THREADS PER BLOCK = 1024')
plt.plot(x,y2, label='THREADS PER BLOCK = 512')
plt.xlabel('2^n : size of the array. n shown on axis')
plt.ylabel('Time Spent(ms)')
plt.title('scaling analysis')
plt.legend()
plt.savefig('task2_t1024_t512.png')
plt.show()
