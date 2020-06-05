'''
BVP SOLUTION
======================================================
Author : Rounak Chatterjee
Date : 05/06/2020
======================================================
There are a number of ways to solve BVPs, but relaxation methods are 
best if the differential equation has a specific form given by:

-y'' + q(x)y = g(x)				--------------------------------(1)
 where a<=x<=b and y(a) = ya y(b) = yb

we follow the method similar to one given in Bulrish and stoer 
section 7.4 Difference Equation where we discretize the equation.
There it says that a unique solution to this equation exists only if 
q(x) for all x >= 0

y'' = (y_(i+1) - 2y_i + y_(i-1))/h^2 where we have divided the interval into n+1 equal
sub intervals i.e h = (b-a)/(n+1) and x_0 = a while x_(n+1) = b.

so we'll get the intermeidate interval points as y_1 to y_n with the 
boundary values as y_0 = ya and y_n+1 = yb.

So if we plug these points in the differential equation we get a set of
n linear equations that can be written in matrix vector form as:

y = Ak where

y = (y_1,...,y_n) k = (g(x_1) + ya/h^2, g_2,...,g_(n-1),g_n+yb/h^2)

Now the matrix A is a Tridiagonal matrix which has the form with 1/h^2
as a common factor, the super and sub diaginal have negative ones while 
the diaginal has form A_(ii) = 2+h^2*q(x_i)

Thus if we solve this set of linear simultaneous equations we can get a set of solution points
for the differential equation

Given Equation in the problem 

y''(x) = 4(y-x)

we can refrom the equation as:

-y''(x) + 4y = 4x

so obviously q(x) = 4 >= 0 for all x in [0,2] and g(x) = 4x

To Solve this problem we take the help of numpy.linalg.solve() function since for this case
the Matrix is a Trigiagonal matrix and the speed of solution will be considerably fast
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
# Range of values
a = 0.0 
b = 1.0
# Boundary Values
y_a = 0.0
y_b = 2.0
n = 99 # Number of solution points 
h = (b-a)/(n+1)
x = np.delete(np.arange(a,b,h),0) # this creates the time mesh points between a and b, i.e excluding a and b
def y(x): # True Solution
    return np.exp(2.0)*(np.exp(4.0)-1)**-1.0*(np.exp(2.0*x)-np.exp(-2.0*x))+x

def q(x): 
    return 4.0
def g(x):
    return 4.0*x

def Create_A(x):
    sup = -1.0*np.ones(len(x)-1,dtype = np.float64)
    sub = -1.0*np.ones(len(x)-1,dtype = np.float64)
    dig = np.zeros(len(x),dtype = np.float64)
    dig[:] = 2.0+q(x[:])*h**2.0
    A = diags((sub,dig,sup),offsets=(-1,0,1),shape=(len(x),len(x)),dtype=np.float64)
    A = A.toarray()*(1/h**2.0) # Converting sparse to Array
    return A
def create_k(x):
    k = np.zeros(len(x),dtype = np.float64)
    k[1:len(x)-2] = g(x[1:len(x)-2])
    k[0] = g(x[0])+y_a/h**2.0
    k[len(x)-1] = g(x[len(x)-1]) + y_b/h**2.0
    return k
# Creating the Two matrices
A = Create_A(x)
k = create_k(x)
sol = np.linalg.solve(A,k) # This does not include the boundaries
# Here we put the boundaries one by one : putting left
x_full = np.insert(x,0,a)
sol = np.insert(sol,0,y_a)
# Putting Right
x_full = np.insert(x_full,len(x_full),b)
sol = np.insert(sol,len(sol),y_b)

#PLotting the Computed Solution Vs The true Solution for 101 points

plt.title("Solution of BVP by Relaxation method",size = 15)
plt.scatter(x_full,sol,marker = 'x',color = 'black',label = "Obtained Numerical Solution")
plt.plot(np.linspace(a,b,1000),y(np.linspace(a,b,1000)),color = '#00FF00',label = "Analytical solution")
plt.grid()
plt.xlabel("y(x)",size = 13)
plt.ylabel("x",size = 13)
plt.legend()
plt.show()

'''
Calculating Error
==================================
Since the error occurs only in the inbetween computed points, we're gonna
compare them only.

err = |y_true - y_comp|/y_true

Each point will have a relative error percentage that we will quote.
FInally we will quote the average error percentage with a variance.
''' 
err = np.zeros(n,dtype = np.float64)
print("Point Index\tTrue Value\t\t\t\tComputed\t\t\t\t\tError\t\t\t\t\tpercentage")
for i in range(n):
	err[i] = np.abs(y(x[i]) - sol[i+1])/y(x[i])
	print("\t",(i+1),"\t",y(x[i]),"\t",sol[i+1],"\t",err[i],"\t",100.0*err[i],"%")
print("mean error percent = ",np.mean(err)*100.0,"%")
print("variance = ",np.var(err)*100.0,"%")



