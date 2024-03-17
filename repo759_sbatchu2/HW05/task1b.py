import matplotlib.pyplot as plt

y = []
y2 = []
y3 = []
x = []

with open('task1_time_m1_t32.log', 'r') as file:
    for line in file:
        y.append(float(line))
with open('task1_time_m2_t32.log', 'r') as file:
    for line in file:
        y2.append(float(line))
with open('task1_time_m3_t32.log', 'r') as file:
    for line in file:
        y3.append(float(line))


for i in range(5,12): #12 not inclusive
    x.append(f"{i}")

plt.figure(1)
plt.plot(x,y)
plt.xlabel('2^n : size of the array. n shown on axis, block_dim : 32')
plt.ylabel('Time Spent(ms)')
plt.title('scaling analysis : matmul_1, datatype : int')
plt.savefig('task1_m1_t32.png')

plt.figure(2)
plt.plot(x,y2)
plt.xlabel('2^n : size of the array. n shown on axis, block_dim : 32')
plt.ylabel('Time Spent(ms)')
plt.title('scaling analysis : matmul_2, datatype : float')
plt.savefig('task1_m2_t32.png')

plt.figure(3)
plt.plot(x,y3)
plt.xlabel('2^n : size of the array. n shown on axis, block_dim : 32')
plt.ylabel('Time Spent(ms)')
plt.title('scaling analysis : matmul_3, datatype : double')
plt.savefig('task1_m2_t32.png')
#
plt.figure(4)
plt.plot(x,y, label='matmul1 Datatype : int')
plt.plot(x,y2, label='matmul2 Datatype : float')
plt.plot(x,y3, label='matmul3 Datatype : double')
plt.xlabel('2^n : size of the array. n shown on axis, block_dim : 32')
plt.ylabel('Time Spent(ms)')
plt.title('scaling analysis')
plt.legend()
plt.savefig('task1_matmul_t32.png')
plt.show()
