'''
SOLVING THE SET OF EQUATIONS
==============================================================================
Author : Rounak Chatterjee
Date : 05/05/2020
==============================================================================
Given this set of simultaneous Equation 1st Order, the best way to solve it is by using 
4th Order Runge Kutta Method. Here we'll use a custom code to better 
handle the vectorial nature of the problem.

In this we denote r = [y1,y2] so r[0] = y1 and r[1] = y2
'''

import numpy as np
import matplotlib.pyplot as plt
h = 0.001 # step value
a = 0.0 #Initial Value
b = 0.5 #Final Value


def f(r,x): #The Vector function 
	return np.array([32.0*r[0]+66.0*r[1]+(2.0/3.0)*x+(2.0/3.0),-66.0*r[0]-133.0*r[1]-(1.0/3.0)*x-(1.0/3.0)])

def next_value(wi,xi): # This Executes a step of The RK Algorithm
	k1 = h*f(wi,xi)
	k2 = h*f(wi+k1/2.0,xi+h/2.0)
	k3 = h*f(wi+k2/2,xi+h/2.0)
	k4 = h*f(wi+k3,xi+h) 
	return (wi+(1.0/6.0)*(k1+2.0*k2+2.0*k3+k4))

n = int((b-a)/h) # total number of steps
x = np.arange(a,b,h)

w = np.zeros(shape=(n,2),dtype=np.float64) # Solution set
w[0]=[(1.0/3.0),(1.0/3.0)] # Initial values

for i in range(n-1):
	w[i+1] = next_value(w[i],a+i*h)

plt.title("Plot of Computed Solutions",size = 15)
plt.plot(x,w[:,0],color = 'blue',label = 'y$_1$(x)')
plt.plot(x,w[:,1],color = 'red',label = 'y$_2$(x)') 
plt.legend()
plt.xlabel("x",size = 13)
plt.ylabel("y(x)",size = 13)
plt.grid()
plt.show()

'''
This Part of Code Produces the graph that justifies that the solution 
obtained is correct. As analytically obtained we see that the function 2y1 + y2 is of form  
 2y1 + y2 = exp(-x)+x

here we try to verify it.
'''
plt.title("Justification Plot",size = 15)
plt.plot(x,2.0*w[:,0]+w[:,1],lw =5 ,color = 'black',label = '2y$_1$(x)+y$_2$(x)')
plt.plot(x,np.exp(-x)+x,color = '#00FF00',label = "Analytical Solution") 
plt.legend()
plt.xlabel("x",size = 13)
plt.ylabel("y(x)",size = 13)
plt.grid()
plt.show()

