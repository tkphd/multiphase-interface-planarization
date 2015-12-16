# Plot the bulk contributions (no gradients) for a selection of
# multiphase-field models. For details, see
# Toth et al., Phys. Rev. B 92 (2015) 184105.

import matplotlib.pyplot as plt
import numpy as np

# interfacial Helmholtz free energy densities
def binary_folch(a, b):
	return np.power(a*(1.0-a),2) + np.power(b*(1.0-b),2)

def ternary_folch(a, b, c):
	return np.power(a*(1.0-a),2) + np.power(b*(1.0-b),2) + np.power(c*(1.0-c),2)

def binary_steinbach(a, b):
	return np.abs(a)*np.abs(b)

def ternary_steinbach(a, b, c):
	return np.abs(a)*np.abs(b) + np.abs(a)*np.abs(c) + np.abs(b)*np.abs(c)		

def binary_toth(a, b):
	return 1.0/12 + (0.25*a**4 - (1.0/3)*a**3) + (0.25*b**4 - (1.0/3)*b**3) + 0.5*a**2*b**2

def ternary_toth(a, b, c):
	return 1.0/12 \
	       + (0.25*a**4 - (1.0/3)*a**3) + (0.25*b**4 - (1.0/3)*b**3) + (0.25*c**4 - (1.0/3)*c**3) \
	       + 0.5*(a**2*b**2 + a**2*c**2 + b**2+c**2)

def binary_moelans(a, b):
	gamma = 1.5
	return 1.0/4 + (0.25*a**4 - 0.5*a**2) + (0.25*b**4 - 0.5*b**2) + gamma*a**2*b**2

def ternary_moelans(a, b, c):
	gamma = 1.5
	return 1.0/4 \
	       + (0.25*a**4 - 0.5*a**2) + (0.25*b**4 - 0.5*b**2) + (0.25*c**4 - 0.5*c**2) \
	       + gamma*(a**2*b**2 + a**2*c**2 + b**2+c**2)

def binary_levitas(a, b):
	return (a + b - 1.0)**2*a**2*b**2

def ternary_levitas(a, b, c):
	return (a + b - 1.0)**2*a**2*b**2 + (a + c - 1.0)**2*a**2*c**2 + (b + c - 1.0)**2*b**2*c**2 \
	       + 0.0625*a**2*b**2*c**2


# Binary system
a = np.linspace(-1,1,500)
b = np.linspace(-1,1,500)

x = np.zeros(len(a)*len(b))
y = np.zeros(len(a)*len(b))
zs = np.zeros(len(a)*len(b))
zt = np.zeros(len(a)*len(b))
zm = np.zeros(len(a)*len(b))
zf = np.zeros(len(a)*len(b))
zl = np.zeros(len(a)*len(b))
zpath = np.zeros(len(a)*len(b))

n=0
for j in np.nditer(b):
	for i in np.nditer(a):
		x[n] = i
		y[n] = j
		zpath[n] = i+j
		zs[n] = binary_steinbach(i, j)
		zt[n] = binary_toth(i, j)
		zm[n] = binary_moelans(i, j)
		zf[n] = binary_folch(i, j)
		zl[n] = binary_levitas(i, j)
		n+=1

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
f.suptitle("MPF Binary Potentials",fontsize=14)
ax1.set_title(r'Folch bulk energy',fontsize=10)
ax1.axis('equal')
confil = ax1.tricontourf(x,y,zf, 96, cmap=plt.cm.get_cmap('coolwarm'))
ax1.tricontour(x,y,zpath, [1.0])
ax2.set_title(r'Steinbach bulk energy',fontsize=10)
ax2.axis('equal')
confil = ax2.tricontourf(x,y,zs, 96, cmap=plt.cm.get_cmap('coolwarm'))
ax2.tricontour(x,y,zpath, [1.0])
ax3.set_title(r'Moelans bulk energy',fontsize=10)
ax3.axis('equal')
confil = ax3.tricontourf(x,y,zm, 96, cmap=plt.cm.get_cmap('coolwarm'))
ax3.tricontour(x,y,zpath, [1.0])
ax4.set_title(r'Toth bulk energy',fontsize=10)
ax4.axis('equal')
confil = ax4.tricontourf(x,y,zt, 96, cmap=plt.cm.get_cmap('coolwarm'))
ax4.tricontour(x,y,zpath, [1.0])
f.savefig('binary.png', dpi=400, bbox_inches='tight')
plt.close()

f,ax = plt.subplots(1,1)
plt.title("Levitas bulk energy")
plt.axis('equal')
confil = ax.tricontourf(x,y,zl, 96, cmap=plt.cm.get_cmap('coolwarm'))
ax.tricontour(x,y,zpath, [1.0,-1.0])
cbar = plt.colorbar(confil)#, format='%.1f')
plt.savefig('Levitas_binary.png', dpi=400, bbox_inches='tight')
plt.close()

#f,ax = plt.subplots(1,1)
#plt.title("Steinbach bulk energy")
#plt.axis('equal')
#confil = ax.tricontourf(x,y,zs, 96, cmap=plt.cm.get_cmap('coolwarm'))
#ax.tricontour(x,y,zpath, [1.0,-1.0])
#cbar = plt.colorbar(confil)#, format='%.1f')
#plt.savefig('Steinbach_binary.png', dpi=400, bbox_inches='tight')
#plt.close()

#f,ax = plt.subplots(1,1)
#plt.title("Toth bulk energy")
#plt.axis('equal')
#confil = ax.tricontourf(x,y,zt, 96, cmap=plt.cm.get_cmap('coolwarm'))
#ax.tricontour(x,y,zpath, [1.0,-1.0])
#cbar = plt.colorbar(confil)#, format='%.1f')
#plt.savefig('Toth_binary.png', dpi=400, bbox_inches='tight')
#plt.close()

#f,ax = plt.subplots(1,1)
#plt.title("Moelans bulk energy")
#plt.axis('equal')
#confil = ax.tricontourf(x,y,zm, 96, cmap=plt.cm.get_cmap('coolwarm'))
#ax.tricontour(x,y,zpath, [1.0,-1.0])
#cbar = plt.colorbar(confil)#, format='%.1f')
#plt.savefig('Moelans_binary.png', dpi=400, bbox_inches='tight')
#plt.close()


# Ternary system
npts = 400
x = np.linspace(0,1.01,npts)
y = np.linspace(0,1.01,npts)

p = []
q = []
zs = []
zt = []
zm = []
zf = []
zl = []
#p = [0,1,0.5]
#q = [0,0,1]
#zs = [ternary_steinbach(1,0,0),ternary_steinbach(0,1,0),ternary_steinbach(0,0,1)]
#zt = [ternary_toth(1,0,0),ternary_toth(0,1,0),ternary_toth(0,0,1)]
#zm = [ternary_moelans(1,0,0),ternary_moelans(0,1,0),ternary_moelans(0,0,1)]

for j in np.nditer(y):
	for i in np.nditer(x):
		c = 2.0*j/np.sqrt(3.0)
		b = (2.0*i-c)/2.0
		a = 1.0 - b - c
		if (a>-0.001 and b>-0.001):
			p.append(i)
			q.append(j)
			zs.append(ternary_steinbach(a,b,c))
			zt.append(ternary_toth(a,b,c))
			zm.append(ternary_moelans(a,b,c))
			zf.append(ternary_folch(a,b,c))
			zl.append(ternary_levitas(a,b,c))

f, ax = plt.subplots(1,1)
plt.title(r'Levitas bulk energy')
plt.axis('equal')
plt.axis('off')
confil = ax.tricontourf(p,q,zl, 96, cmap=plt.cm.get_cmap('coolwarm'))
cbar = plt.colorbar(confil)#, format='%.1f')
plt.savefig('Levitas.png', dpi=400, bbox_inches='tight')
plt.close()

#f, ax = plt.subplots(1,1)
#plt.title(r'Steinbach bulk energy')
#plt.axis('equal')
#plt.axis('off')
#confil = ax.tricontourf(p,q,zs, 96, cmap=plt.cm.get_cmap('coolwarm'))
#cbar = plt.colorbar(confil)#, format='%.1f')
#plt.savefig('Steinbach.png', dpi=400, bbox_inches='tight')
#plt.close()

#f,ax = plt.subplots(1,1)
#plt.title(r'Toth bulk energy')
#plt.axis('equal')
#plt.axis('off')
#confil = ax.tricontourf(p,q,zt, 96, cmap=plt.cm.get_cmap('coolwarm'))
#cbar = plt.colorbar(confil)#, format='%.1f')
#plt.savefig('Toth.png', dpi=400, bbox_inches='tight')
#plt.close()

#f,ax = plt.subplots(1,1)
#plt.title(r'Moelans bulk energy')
#plt.axis('equal')
#plt.axis('off')
#confil = ax.tricontourf(p,q,zm, 96, cmap=plt.cm.get_cmap('coolwarm'))
#cbar = plt.colorbar(confil)#, format='%.1f')
#plt.savefig('Moelans.png', dpi=400, bbox_inches='tight')
#plt.close()

#f,ax = plt.subplots(1,1)
#plt.title(r'Folch bulk energy')
#plt.axis('equal')
#plt.axis('off')
#confil = ax.tricontourf(p,q,zf, 96, cmap=plt.cm.get_cmap('coolwarm'))
#cbar = plt.colorbar(confil)#, format='%.1f')
#plt.savefig('Folch.png', dpi=400, bbox_inches='tight')
#plt.close()

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
f.suptitle("MPF Ternary Simplices",fontsize=14)
ax1.set_title(r'Folch bulk energy',fontsize=10)
ax1.axis('equal')
ax1.axis('off')
confil = ax1.tricontourf(p,q,zf, 96, cmap=plt.cm.get_cmap('coolwarm'))
ax2.set_title(r'Steinbach bulk energy',fontsize=10)
ax2.axis('equal')
ax2.axis('off')
confil = ax2.tricontourf(p,q,zs, 96, cmap=plt.cm.get_cmap('coolwarm'))
ax3.set_title(r'Moelans bulk energy',fontsize=10)
ax3.axis('equal')
ax3.axis('off')
confil = ax3.tricontourf(p,q,zm, 96, cmap=plt.cm.get_cmap('coolwarm'))
ax4.set_title(r'Toth bulk energy',fontsize=10)
ax4.axis('equal')
ax4.axis('off')
confil = ax4.tricontourf(p,q,zt, 96, cmap=plt.cm.get_cmap('coolwarm'))
f.savefig('simplex.png', dpi=400, bbox_inches='tight')
plt.close()
