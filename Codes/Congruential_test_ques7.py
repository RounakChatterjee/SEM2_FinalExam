import numpy as np
a = 1103515245
c = 12345
m = 2e31
x0 = 1 # initial seed
runs = np.int64(100000000)
x = x0
for i in range(runs):
	x = (a*x+c)%m
	if(x == x0):
		print("Found x0 again at itteration = ",i)
		exit()
print("x0 not found in the itteration")
