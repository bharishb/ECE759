import matplotlib.pyplot as plt

y = []
y_same_mem = []
x = []

with open('task1_time.log', 'r') as file:
    for line in file:
        y.append(float(line))
#with open('task1_time_same_mem.log', 'r') as file:
#    for line in file:
#        y_same_mem.append(float(line))


for i in range(10,31): #21 not inclusive
    x.append(f"{i}")

plt.figure(1)
plt.plot(x,y)
plt.xlabel('2^n : size of the array. n shown on axis')
plt.ylabel('Time Spent in Scan function(ms)')
plt.title('Scan function scaling analysis')
plt.savefig('task1.png')

#plt.figure(2)
#plt.plot(x,y_same_mem)
#plt.xlabel('n : size of the array')
#plt.ylabel('Time Spent in Scan function(ms)')
#plt.title('Scan function scaling analysis : Same Input/Output Memory')
#plt.savefig('task1_same_mem.png')
#
#plt.figure(3)
#plt.plot(x,y, label='Different Input Output Mem')
#plt.plot(x,y_same_mem, label='Same Input Output Mem')
#plt.xlabel('n : size of the array')
#plt.ylabel('Time Spent in Scan function(ms)')
#plt.title('Scan function scaling analysis : With/Without same memory')
#plt.legend()
#plt.savefig('task1_compare.png')
plt.show()
