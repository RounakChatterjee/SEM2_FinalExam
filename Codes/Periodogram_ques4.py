'''
PERIODOGRAM
===============================================================
Author : Rounak Chatterjee
Date : 06/05/2020
===============================================================
This Program computes the periodogram fror 1024 randomly generated data from
a uniform diistribution. We will use all numpy packages like numpy.random
and numpy.fft to do what is necessary. Finally the periodogrm will be constructed out of 
Bartlett averaging method.
'''
import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt

def calc_pow_spec(data,binned = False,bin_size = 0):# This Function can calculate power spectrum for both binned and unbinned data
	if(binned == False ): # for Unbinned Data
		k = np.fft.fftshift(np.fft.fftfreq(len(data)))
		dft = np.fft.fftshift(np.fft.fft(data,norm = 'ortho'))
		spec = (1.0/len(data))*np.absolute(dft)**2.0
		return [k,spec]
	else: # for Binned Data
		if(len(data)%bin_size == 0):
			spec = np.zeros(len(data)//bin_size,dtype = np.float64)
			data_bin = np.zeros(shape = (len(data)//bin_size,bin_size),dtype = np.float64)
		else:
			spec = np.zeros(len(data)//bin_size+1,dtype = np.float64)
			data_bin = np.zeros(shape = (len(data)//bin_size+1,bin_size),dtype = np.float64)
			k = 0
			for i in range(len(spec)):
				for j in range(bin_size):
					data_bin[i][j] = data[k]
					k = k+1
					if(k == len(data)):
						break
			for i in range(len(spec)):
				dft = np.fft.fftshift(np.fft.fft(data_bin[i],norm = 'ortho'))
				spec[i] = np.mean(np.absolute(dft)**2.0)
			return spec
		

data_size = 1024
rand_data = np.random.rand(data_size)# we create from distribution f(x) = 1.0
pow_spec = calc_pow_spec(rand_data)
print("The Minimum and Maximum  value of k(dft frequencies) are:\nMin value = ",min(pow_spec[0]),"\n Max Value = ",max(pow_spec[0]))
print("The Minimum and Maximum 'absolute' value of k(dft frequencies) are:\nMin value = ",min(np.abs(pow_spec[0])),"\n Max Value = ",max(np.abs(pow_spec[0])))
binned_pow_spec = calc_pow_spec(rand_data,binned = True,bin_size = 5)

fig = plt.figure(constrained_layout  = True)
spec = fig.add_gridspec(2,2)
d_ft = fig.add_subplot(spec[0,1])
d_ft.set_title("DFT of Data",size = 13)
dft = np.fft.fftshift(np.fft.fft(rand_data,norm = 'ortho'))
k = np.fft.fftshift(np.fft.fftfreq(len(rand_data)))
d_ft.plot(k,dft,color = 'green',label = 'DFT')
d_ft.set_xlabel("frequencies")
d_ft.set_ylabel("DFT Values")
d_ft.legend()
d_ft.grid()

dat = fig.add_subplot(spec[0,0])
dat.set_title("Data",size = 13)
dat.scatter(np.linspace(1,len(rand_data),len(rand_data)),rand_data)
dat.set_xlabel("Data Label")
dat.set_ylabel("Random Data")
dat.grid()

ft = fig.add_subplot(spec[1,0])
ft.set_title("Normal periodogram",size = 13)
ft.plot(pow_spec[0],pow_spec[1],color = 'red',label = 'Periodogram')
ft.set_xlabel("frequencies(Hz)")
ft.set_ylabel("Periodogram")
ft.legend()
ft.grid()

bin_ft = fig.add_subplot(spec[1,1])
bin_ft.set_title("Binned Periodogram(Bartlett)",size = 13)
bin_ft.stem(np.linspace(0,len(binned_pow_spec),len(binned_pow_spec)),binned_pow_spec,markerfmt = ('o','black'),linefmt = ('-','red'),label = "5 point bins")
bin_ft.set_xlabel("frequency bins")
bin_ft.set_ylabel("average periodogram value")
bin_ft.legend()
bin_ft.grid()


plt.show()

