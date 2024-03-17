import matplotlib.pyplot as plt

y = []
y2 = []
x = []

with open('task3_time.log', 'r') as file:
    for line in file:
        y.append(float(line))
with open('task3_time_t16.log', 'r') as file:
    for line in file:
        y2.append(float(line))


for i in range(10,30): #30 not inclusive
    x.append(f"{i}")

plt.figure(1)
plt.plot(x,y)
plt.xlabel('2^n : size of the array. n shown on axis')
plt.ylabel('Time Spent(ms)')
plt.title('vscale scaling analysis : THREADS PER BLOCK = 512')
plt.savefig('task3_t512.png')

plt.figure(2)
plt.plot(x,y2)
plt.xlabel('2^n : size of the array. n shown on axis')
plt.ylabel('Time Spent(ms)')
plt.title('vscale scaling analysis : THREADS PER BLOCK = 16')
plt.savefig('task3_t16.png')
#
plt.figure(3)
plt.plot(x,y, label='THREADS PER BLOCK = 512')
plt.plot(x,y2, label='THREADS PER BLOCK = 16')
plt.xlabel('2^n : size of the array. n shown on axis')
plt.ylabel('Time Spent(ms)')
plt.title('vscale scaling analysis')
plt.legend()
plt.savefig('task3_t16_t512.png')
plt.show()
