import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns

def gaussianRandomField(sdf, binary = False, number_of_points = 1e3):
	# generate GRF using cahn's wave form approach

	kmax = np.pi
	pixel = 2*(len(sdf) - 1)
	dk = 2*kmax/pixel
	k = np.linspace(0, kmax, len(sdf))
	rho = sdf

	# calculate pdf of k
	P_k = np.multiply(k, rho)
	P_k = [max(p, 0) for p in P_k]
	# normalize P_k
	P_k = P_k / sum(P_k) / dk

	N = number_of_points
	CP_k = np.cumsum(np.multiply(P_k,dk))

	num_inerp_points = 1e5
	invP_k = np.interp(np.linspace(0, 1, num_inerp_points),CP_k, k)
	
	ki = invP_k[np.random.randint(0, num_inerp_points, (N, 1))]
	
	ph = np.pi*np.random.rand(N,1)
	kiV = np.concatenate((np.cos(ph),np.sin(ph)), axis = 1)
	
	del ph
	
	phi = 2 * np.pi * np.random.rand(N, 1)
	[rx,ry] = np.meshgrid(np.linspace(1,pixel, num = pixel), np.linspace(1, pixel, num = pixel))
	
	r = np.concatenate((rx.reshape(1, pixel**2), ry.reshape(1, pixel**2)), axis=0)
	ki = np.matlib.repmat(ki, 1, pixel**2)
	phi = np.matlib.repmat(phi, 1, pixel**2)
	
	yr = np.sqrt(2.0/N)*np.sum(np.cos(np.multiply(ki, np.dot(kiV, r)) + phi), axis = 0)

	del phi, r, ki, kiV

	image = yr.reshape((pixel, pixel))

	return image


def sdf_uniform(start, end, range, num = 101):
	# generate a uniform spectral density function
	a, b = start, end
	x = np.linspace(range[0], range[1], num)
	
	sdf = np.piecewise(x, [x < start, (x >= start)&(x <= end), x > end], [lambda x: 0, lambda x: 1, lambda x: 0])
	return sdf / sum(sdf)