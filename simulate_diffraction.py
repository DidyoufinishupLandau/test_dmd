import numpy as np
import matplotlib.pyplot as plt
import math
def sinc(x):
    return np.sin(x)/x
def convolution(angle,wavelength,slit_separation):
    x = np.pi * slit_separation * np.sin(angle) / (wavelength)
    return sinc(x)*(1+np.exp(1j*x)+np.exp(1j*100*x)+np.exp(1j*1000*x)+np.exp(1j*3*x)+np.exp(1j*10*x))
x = np.linspace(-np.pi/2,np.pi/2,100)
y = convolution(x, 800*10**-9, 10**-6)
y = y*np.conj(y)
y = np.sqrt(y)
plt.plot(x, y)
#plt.plot(x, sinc(x))
plt.show()