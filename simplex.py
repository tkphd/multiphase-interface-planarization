# -*- coding: utf-8 -*-
# Plot the bulk contributions (no gradients) for a selection of
# multiphase-field models. For details, see
# Toth et al., Phys. Rev. B 92 (2015) 184105.

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

Titles = (u'Folch bulk energy',u'Steinbach bulk energy',u'Moelans bulk energy',u'Tóth bulk energy')

# interfacial Helmholtz free energy densities
def binary_folch(a, b):
	return np.power(a*(1.0-a),2) + np.power(b*(1.0-b),2)

def ternary_folch(a, b, c):
	return np.power(a*(1.0-a),2) + np.power(b*(1.0-b),2) + np.power(c*(1.0-c),2)

def binary_steinbach(a, b):
	return np.abs(a)*np.abs(b)

def ternary_steinbach(a, b, c):
	return np.abs(a)*np.abs(b) + np.abs(a)*np.abs(c) + np.abs(b)*np.abs(c)		

def binary_moelans(a, b):
	gamma = 1.5
	return 1.0/4 + (0.25*a**4 - 0.5*a**2) + (0.25*b**4 - 0.5*b**2) + gamma*a**2*b**2

def ternary_moelans(a, b, c):
	gamma = 1.5
	return 1.0/4 \
	       + (0.25*a**4 - 0.5*a**2) + (0.25*b**4 - 0.5*b**2) + (0.25*c**4 - 0.5*c**2) \
	       + gamma*(a**2*b**2 + a**2*c**2 + b**2+c**2)

def binary_toth(a, b):
	return 1.0/12 + (0.25*a**4 - (1.0/3)*a**3) + (0.25*b**4 - (1.0/3)*b**3) + 0.5*a**2*b**2

def ternary_toth(a, b, c):
	return 1.0/12 \
	       + (0.25*a**4 - (1.0/3)*a**3) + (0.25*b**4 - (1.0/3)*b**3) + (0.25*c**4 - (1.0/3)*c**3) \
	       + 0.5*(a**2*b**2 + a**2*c**2 + b**2+c**2)

def binary_levitas(a, b):
	l=2
	return (a + b - 1.0)**2*np.power(a,l)*np.power(b,l)

def ternary_levitas(a, b, c):
	l=2
	return (a + b - 1.0)**2*np.power(a,l)*np.power(b,l) + (a + c - 1.0)**2*np.power(a,l)*np.power(c,l) + (b + c - 1.0)**2*np.power(b,l)*np.power(c,l) \
	       + 0.0625*a**2*b**2*c**2

# Binary system
span = (-1,1.1)
a = np.linspace(span[0],span[1],400)
b = np.linspace(span[0],span[1],400)

x = np.zeros(len(a)*len(b))
y = np.zeros(len(a)*len(b))
z = np.ndarray(shape=(5,len(a)*len(b)), dtype=float)
zpath = np.zeros(len(a)*len(b))

sqx = np.array([0,1,1,0,0])
sqy = np.array([0,0,1,1,0])

n=0
for j in np.nditer(b):
	for i in np.nditer(a):
		x[n] = i
		y[n] = j
		zpath[n] = i+j
		z[0][n] = binary_folch(i, j)
		z[1][n] = binary_steinbach(i, j)
		z[2][n] = binary_moelans(i, j)
		z[3][n] = binary_toth(i, j)
		z[4][n] = binary_levitas(i, j)
		n+=1

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row')
f.suptitle("MPF Binary Potentials",fontsize=14)
n=0
for ax in axarr.reshape(-1):
	ax.set_title(Titles[n],fontsize=10)
	ax.axis('equal')
	ax.set_xlim(span)
	ax.set_ylim(span)
	ax.axis('off')
	confil = ax.tricontourf(x,y,z[n], 96, cmap=plt.cm.get_cmap('coolwarm'))
	ax.tricontour(x,y,zpath, [1.0])
	ax.plot(sqx,sqy, linestyle=':', color='w')
	n+=1
f.savefig('binary.png', dpi=400, bbox_inches='tight')
plt.close()

f,ax = plt.subplots(1,1)
plt.title("Levitas bulk energy")
plt.axis('equal')
confil = ax.tricontourf(x,y,z[4], 96, cmap=plt.cm.get_cmap('coolwarm'))
ax.tricontour(x,y,zpath, [1.0,-1.0])
cbar = plt.colorbar(confil)#, format='%.1f')
ax.plot(sqx,sqy, linestyle=':', color='w')
plt.savefig('Levitas_binary.png', dpi=400, bbox_inches='tight')
plt.close()

# Ternary system
npts = 100
span = (-0.25,1.25)
x = np.linspace(span[0],span[1],npts)
y = np.linspace(span[0],span[1],npts)
z = np.ndarray(shape=(5,len(x)*len(y)), dtype=float)

trix = np.array([0,1,0.5,0])
triy = np.array([0,0,np.sqrt(3)/2,0])
offx = 0.5*(trix - 1)
offy = 0.5*(triy - 1)

p = np.zeros(len(x)*len(y))
q = np.zeros(len(x)*len(y))

n=0
for j in np.nditer(y):
	for i in np.nditer(x):
		c = 2.0*j/np.sqrt(3.0)
		b = (2.0*i-c)/2.0
		a = 1.0 - b - c
		p[n]=i
		q[n]=j
		z[0][n] = ternary_folch(a,b,c)
		z[1][n] = ternary_steinbach(a,b,c)
		z[2][n] = ternary_toth(a,b,c)
		z[3][n] = ternary_moelans(a,b,c)
		z[4][n] = ternary_levitas(a,b,c)
		n+=1

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row')
f.suptitle("MPF Ternary Potentials",fontsize=14)
n=0
for ax in axarr.reshape(-1):
	ax.set_title(Titles[n],fontsize=10)
	ax.axis('equal')
	ax.set_xlim(span)
	ax.set_ylim(span)
	ax.axis('off')
	ax.tricontourf(p,q,z[n], 96, cmap=plt.cm.get_cmap('coolwarm'),)#)
	ax.plot(trix,triy, linestyle=':', color='w')
	n+=1
f.savefig('ternary.png', dpi=400, bbox_inches='tight')
plt.close()

f, ax = plt.subplots(1,1)
plt.title(r'Levitas bulk energy')
plt.axis('equal')
plt.axis('off')
confil = ax.tricontourf(p,q,z[4], 96, cmap=plt.cm.get_cmap('coolwarm'),norm=LogNorm())
cbar = plt.colorbar(confil)#, format='%.1f')
plt.savefig('Levitas.png', dpi=400, bbox_inches='tight')
plt.close()

