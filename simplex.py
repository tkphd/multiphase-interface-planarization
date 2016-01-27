# -*- coding: utf-8 -*-
# Plot the bulk contributions (no gradients) for a selection of multiphase-field models.

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
plt.rcParams['figure.figsize'] = (12.0, 8.0)

Titles = (u'Folch',u'Steinbach',u'Chen',u'Moelans',u'Tóth',u'Levitas')
nmodels = len(Titles)

# Folch and Plapp. Phys. Rev. E 72 (2005) 011602. Eqn. 3.14.
def binary_folch(a, b):
	return np.power(a*(1.0-a),2) + np.power(b*(1.0-b),2)

def ternary_folch(a, b, c):
	return np.power(a*(1.0-a),2) + np.power(b*(1.0-b),2) + np.power(c*(1.0-c),2)

# Steinbach and Pezzolla. Physica D 134 (1999) 385-393. Eqn. 10.
def binary_steinbach(a, b):
	return 0.5*np.abs(a)*np.abs(b)

def ternary_steinbach(a, b, c):
	return 0.5*np.abs(a)*np.abs(b) + 0.5*np.abs(a)*np.abs(c) + 0.5*np.abs(b)*np.abs(c)		

# Chen and Yang. Phys. Rev. B 50 (1994) 15752. Eqn. 1.
def binary_chen(a, b):
	return (0.25*a**4 - 0.5*a**2) + (0.25*b**4 - 0.5*b**2) + 2.0*a**2*b**2

def ternary_chen(a, b, c):
	return (0.25*a**4 - 0.5*a**2) + (0.25*b**4 - 0.5*b**2) + (0.25*c**4 - 0.5*c**2)\
	       + 2.0*a**2*b**2 + 2.0*a**2*c**2 + 2.0*b**2*c**2

# Moelans, Blanpain, and Wollants. Pyhs. Rev. B 78 (2008) 024113. Eqn. 2.
def binary_moelans(a, b):
	gamma = 1.5
	return 1.0/4 + (0.25*a**4 - 0.5*a**2) + (0.25*b**4 - 0.5*b**2) + gamma*a**2*b**2

def ternary_moelans(a, b, c):
	gamma = 1.5
	return 1.0/4 \
	       + (0.25*a**4 - 0.5*a**2) + (0.25*b**4 - 0.5*b**2) + (0.25*c**4 - 0.5*c**2) \
	       + gamma*(a**2*b**2 + a**2*c**2 + b**2+c**2)

# Toth, Pusztai, and Granasy. Phys. Rev. B 92 (2015) 184105. Eqn. 30.
def binary_toth(a, b):
	return 1.0/12 + (0.25*a**4 - (1.0/3)*a**3) + (0.25*b**4 - (1.0/3)*b**3) + 0.5*a**2*b**2

def ternary_toth(a, b, c):
	return 1.0/12 \
	       + (0.25*a**4 - (1.0/3)*a**3) + (0.25*b**4 - (1.0/3)*b**3) + (0.25*c**4 - (1.0/3)*c**3) \
	       + 0.5*(a**2*b**2 + a**2*c**2 + b**2+c**2)

# Levitas and Roy. Phys. Rev. B 91 (2015) 174109. Eqn. 4.
def binary_levitas(a, b):
	l=2
	return (a + b - 1.0)**2*np.power(a,l)*np.power(b,l)

def ternary_levitas(a, b, c):
	l=2
	return   (a + b - 1.0)**2*np.power(a,l)*np.power(b,l) \
	       + (a + c - 1.0)**2*np.power(a,l)*np.power(c,l) \
	       + (b + c - 1.0)**2*np.power(b,l)*np.power(c,l) \
	       + 0.0625*a**2*b**2*c**2


# Binary system
span = (-1.05,1.05)
a = np.linspace(span[0],span[1],400)
b = np.linspace(span[0],span[1],400)

x = np.zeros(len(a)*len(b))
y = np.zeros(len(a)*len(b))
z = np.ndarray(shape=(nmodels,len(a)*len(b)), dtype=float)
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
		z[2][n] = binary_chen(i, j)
		z[3][n] = binary_moelans(i, j)
		z[4][n] = binary_toth(i, j)
		z[5][n] = binary_levitas(i, j)
		n+=1

f, axarr = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row')
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
plt.figtext(0.5, 0.0625, r'White boxes enclose $x,y\in[0,1]$. Black pathways constrain $\phi_\alpha+\phi_\beta=1$.', ha='center', fontsize=8)
f.savefig('binary.png', dpi=400, bbox_inches='tight')
plt.close()

# Ternary system
npts = 200
span = (-0.05,1.05)
yspan = (-0.15,0.95)
x = np.linspace(span[0],span[1],npts)
y = np.linspace(yspan[0],yspan[1],npts)
z = np.ndarray(shape=(nmodels,len(x)*len(y)), dtype=float)

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
		z[2][n] = ternary_chen(a,b,c)
		z[3][n] = np.min((ternary_moelans(a,b,c),ternary_moelans(b,a,c),ternary_moelans(b,c,a),
		                  ternary_moelans(a,c,b),ternary_moelans(c,a,b),ternary_moelans(c,b,a)))
		z[4][n] = np.min((ternary_toth(a,b,c),ternary_toth(b,a,c),ternary_toth(b,c,a),
		                  ternary_toth(a,c,b),ternary_toth(c,a,b),ternary_toth(c,b,a)))
		z[5][n] = ternary_levitas(a,b,c)
		n+=1

f, axarr = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row')
f.suptitle("MPF Ternary Potentials",fontsize=14)
n=0
for ax in axarr.reshape(-1):
	ax.set_title(Titles[n],fontsize=10)
	ax.axis('equal')
	ax.set_xlim(span)
	ax.set_ylim(yspan)
	ax.axis('off')
	ax.tricontourf(p,q,z[n], 96, cmap=plt.cm.get_cmap('coolwarm'))
	ax.plot(trix,triy, linestyle=':', color='w')
	n+=1
plt.figtext(0.5, 0.0625, r'White triangles enclose Gibbs simplex, $\phi_\alpha+\phi_\beta+\phi_\gamma=1$.', ha='center', fontsize=8)
f.savefig('ternary.png', dpi=400, bbox_inches='tight')
plt.close()

