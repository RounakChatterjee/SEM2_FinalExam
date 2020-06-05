'''
FOURIER TRANSFORM OF BOX FUNCTION
=========================================================
Author : Rounak Chatterjee
Date : 05/06/2020
==========================================================
To Do the fourier transform w will use package numpy.fft 
that contains all the relevant details to do the computation.
The logic beind the computation of discrete fourier transform is:

If {f(x_p)} are the samples of the function at sample points {x_p}
where if x_min and x_max are the two end points, then

d = (x_min-x_max)/(n-1), where n is number of sampled points , hence

x_p = x_min + p.d

thus 

f_ft(k_q) = sqrt(n/2*pi)*d*exp(-2*pi*i*k_q*x_min)*DFT[{f(x_p)}]_q

where DFT[{f(x_p)}]_q is the qth component of the DFT of the sampled points
{f(x_p)} and we have considered the frequencies as k_q = q/(nd), in the same
convention as numpy does to make easy computation. thus the original frequency
components are k_q = 2*Pi*q/(nd), where q varies from 0 to (n-1)

we will use the normalized fft along with this scheme to compute the FT
of the given function

Since the Sampling rate depends on x_min, x_max, n
we will make the sampling rate vary by varrying them.
'''
import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
def f(x): #Given sinc(x) function, can be converted into anythin generic 
	box = np.zeros(len(x),dtype = np.float64)
	for i in range(len(x)):
		if(-1.0<x[i] and x[i]<1.0):
			box[i] = 1.0
		else:
			box[i] = 0.0
	return box

def find_ft(x_min,x_max,n):# Retruns frequecies, ft_values and sampling rate
	x = np.linspace(x_min,x_max,n) 
	d = (x_max-x_min)/(n-1) # Sampling Rate
	dft_f = ft.fft(f(x),norm = 'ortho')
	k =  ft.fftfreq(n,d = d)
	k = 2*np.pi*k
	phi = np.exp(-1.0j*k*x_min)
	ft_f = np.sqrt(n/(2.0*np.pi))*d*phi*dft_f
	ft_f = ft.fftshift(ft_f)
	k = ft.fftshift(k)
	return [k,ft_f,d]


fig = plt.figure(constrained_layout = True)
spec = fig.add_gridspec(2,2)
fig.suptitle("Fourier transform of Box function for various Sampling rates",size = 13)
p1 = fig.add_subplot(spec[0,0])
p1.set_title("Configaration space",size = 12)
p1.plot(np.linspace(-1.5,1.5,1000),f(np.linspace(-1.5,1.5,1000)))
p1.set_xlabel("x",)
p1.set_ylabel("f(x)")
p1.grid()

p2 = fig.add_subplot(spec[0,1])
ft_f = find_ft(-2.0,2.0,512)
p2.set_title("x$_{min}$ = -2.0, x$_{max}$ = 2.0, n = 512")
p2.set_xlabel("k")
p2.set_ylabel("FT(f(x))")
p2.set_xlim([-50.0,50.0])
s = "Sampling Rate = "+str(np.round(ft_f[2],decimals = 5))
p2.plot(ft_f[0],ft_f[1].real,'.-',color='blue',label = s)
p2.legend()
p2.grid()

p3 = fig.add_subplot(spec[1,0])
ft_f = find_ft(-3.0,3.0,1024)
p3.set_title("x$_{min}$ = -3.0, x$_{max}$ = 3.0, n = 1024")
p3.set_xlabel("k")
p3.set_ylabel("FT(f(x))")
p3.set_xlim([-50.0,50.0])
s = "Sampling Rate = "+str(np.round(ft_f[2],decimals = 5))
p3.plot(ft_f[0],ft_f[1].real,'.-',color='red',label = s)
p3.legend()
p3.grid()


p4 = fig.add_subplot(spec[1,1])
ft_f = find_ft(-5.0,5.0,2048)
p4.set_title("x$_{min}$ = -5.0, x$_{max}$ = 5.0, n = 2048")
p4.set_xlabel("k")
p4.set_ylabel("FT(f(x))")
p4.set_xlim([-50.0,50.0])
s = "Sampling Rate = "+str(np.round(ft_f[2],decimals = 5))
p4.plot(ft_f[0],ft_f[1].real,'.-',color='green',label = s)
p4.legend()
p4.grid()
plt.show()
