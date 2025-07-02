# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:40:31 2023

@author: sz3119
"""

import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import time
from scipy.optimize import curve_fit
from BMfunctions import *
from matplotlib.colors import CenteredNorm

N = 1
ky = (2*np.pi)/(2500*km)

#%% create grids
X = np.linspace(-2500,2500,1000)
Y = np.linspace(-2500,2500,1000)
xx,yy = np.meshgrid(X,Y) # ionosphere grid

Ym = np.linspace(-2500,2500,1000)
Z = np.linspace(0,25,1000) # RE
yym,zz = np.meshgrid(Ym,Z) # magnetopause gird
#%% get potentials
sigmaP_list = [1,3,5,7,9]
for sigmaP in sigmaP_list:
    t1 = time.time()
    psi_list = potential(xx*km,yy*km,ky,N,sigmaP) 
    np.savetxt('psi_%sS_1kx1k.txt'%(sigmaP),psi_list)
    t2 = time.time()
    print('sigmaP = %sS done, time used:'%(sigmaP),t2-t1)
#%% rms of jp in x direction
sigmaP_list = [1,3,5,7,9]
rms_max = []

# files = glob.glob('*.txt')
files = glob.glob('1M\*.txt')
for i in range(len(files)):
    t1 = time.time()
    
    psi_list = np.loadtxt(files[i])
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N,sigmaP_list[i])
    jpx,jpy = pedersen(Ex,Ey,sigmaP_list[i])
    rms = np.sqrt(np.trapz((jpx**2+jpy**2).transpose(),Y*km))
    rms_max.append(np.max(rms))
    
    plt.figure('rms_jp_x')
    plt.plot(X,rms,label='$\Sigma_{P}=$'+str(sigmaP_list[i]))
    
    t2 = time.time()
    print('sigmaP = %sS done, time used:'%(sigmaP_list[i]),t2-t1)

plt.xlabel('$x(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.title('RMS magnitude of Pedersen current along x-axis')
plt.legend()
plt.grid()
plt.show()

for i in range(len(rms_max)):
    print(rms_max[i]/rms_max[-1])
  
print()

for i in range(len(rms_max)):
    print((rms_max[i]/rms_max[-1])**2)
#%% rms of jp in y direction
sigmaP_list = [1,3,5,7,9]
rms_max = []

# files = glob.glob('*.txt')
files = glob.glob('1M\*.txt')
for i in range(len(files)):
    t1 = time.time()
    
    psi_list = np.loadtxt(files[i])
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N,sigmaP_list[i])
    jpx,jpy = pedersen(Ex,Ey,sigmaP_list[i])
    rms = np.sqrt(np.trapz((jpx**2+jpy**2),X*km))
    rms_max.append(np.max(rms))
    
    plt.figure('rms_jp_y')
    plt.plot(Y,rms,label='$\Sigma_{P}=$'+str(sigmaP_list[i]))
    
    t2 = time.time()
    print('sigmaP = %sS done, time used:'%(sigmaP_list[i]),t2-t1)

plt.xlabel('$y(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.title('RMS magnitude of Pedersen current along y-axis')
plt.legend()
plt.grid()
plt.show()

for i in range(len(rms_max)):
    print(rms_max[i]/rms_max[-1])
  
print()

for i in range(len(rms_max)):
    print((rms_max[i]/rms_max[-1])**2)
#%% rms of E in x direction
sigmaP_list = [1,3,5,7,9]
rms_max = []

# files = glob.glob('*.txt')
files = glob.glob('1M\*.txt')
for i in range(len(files)):
    t1 = time.time()
    
    psi_list = np.loadtxt(files[i])
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N,sigmaP_list[i])
    rms = np.sqrt(np.trapz((Ex**2+Ey**2).transpose(),Y*km))
    rms_max.append(np.max(rms))
    
    plt.figure('rms_E_x')
    plt.plot(X,rms,label='$\Sigma_{P}=%sS$'%(sigmaP_list[i]))
    
    t2 = time.time()
    print('sigmaP = %sS done, time used:'%(sigmaP_list[i]),t2-t1)

plt.xlabel('$x(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.title('RMS magnitude of electric field along x-axis')
plt.legend()
plt.grid()
plt.show()

for i in range(len(rms_max)):
    print(rms_max[i]/rms_max[-1])
  
print()

for i in range(len(rms_max)):
    print((rms_max[i]/rms_max[-1])**2)
#%% rms of E in y direction
sigmaP_list = [1,3,5,7,9]
rms_max = []

# files = glob.glob('*.txt')
files = glob.glob('1M\*.txt')
for i in range(len(files)):
    t1 = time.time()
    
    psi_list = np.loadtxt(files[i])
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N,sigmaP_list[i])
    rms = np.sqrt(np.trapz((Ex**2+Ey**2),X*km))
    rms_max.append(np.max(rms))
    
    plt.figure('rms_E_y')
    plt.plot(Y,rms,label='$\Sigma_{P}=%sS$'%(sigmaP_list[i]))
    
    t2 = time.time()
    print('sigmaP = %sS done, time used:'%(sigmaP_list[i]),t2-t1)

plt.xlabel('$y(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.title('RMS magnitude of electric field along y-axis')
plt.legend()
plt.grid()
plt.show()

for i in range(len(rms_max)):
    print(rms_max[i]/rms_max[-1])
  
print()

for i in range(len(rms_max)):
    print((rms_max[i]/rms_max[-1])**2)
#%% plot scaling relation
rms_max_plot = rms_max/rms_max[0] 
B = np.linspace(min(sigmaP_list),max(sigmaP_list),1000)

def fit_func(x,a):
    return a/x

initial_guess = [1]
fit = curve_fit(fit_func,sigmaP_list,rms_max_plot,initial_guess,maxfev=1000000)
data_fit = fit_func(B,*fit[0])

plt.figure('rms_peaks')
plt.grid()
plt.plot(sigmaP_list,rms_max_plot,'o',label='Maximum')
plt.plot(B,data_fit,label='Fit = $1/\Sigma_{P}$')
plt.legend()
plt.xlabel('$\Sigma_{P}$(S)')
plt.ylabel('Normalised magnitude')
plt.title('Relation between maximum of RMS(E) and $\Sigma_{P}$')
plt.xticks(sigmaP_list)
plt.show()
print(fit[0][0])
#%% rms of jh in x direction
sigmaH_list = [4,6,5,3.5,2.25]
sigmaP_list = [1,3,5,7,9]

rms_max = []

# files = glob.glob('*.txt')
files = glob.glob('1M\*.txt')
for i in range(len(files)):
    t1 = time.time()
    
    psi_list = np.loadtxt(files[i])
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N,sigmaP_list[i])
    jhx,jhy = hall(Ex,Ey,sigmaH_list[i])
    rms = np.sqrt(np.trapz((jhx**2+jhy**2).transpose(),Y*km))
    rms_max.append(np.max(rms))
    
    plt.figure('rms_jh_x',dpi=150)
    plt.plot(X,rms,label='$\Sigma_{H}/\Sigma_{P}=$'+str(sigmaH_list[i]/sigmaP_list[i]))
    # plt.plot(ky*X,rms*sigmaP_list[i]/sigmaH_list[i],label='$\Sigma_{H}/\Sigma_{P}=$'+str(sigmaH_list[i]/sigmaP_list[i]))
    
    t2 = time.time()
    print('sigmaP = %sS done, time used:'%(sigmaP_list[i]),t2-t1)

plt.xlabel('$x(km)$')
# plt.xticks(np.arange(-2500,2501,500))
plt.title('RMS magnitude of Hall current along x-axis')
plt.legend()
plt.grid()
plt.show()

for i in range(len(rms_max)):
    print(rms_max[i]/rms_max[-1])
  
print()

for i in range(len(rms_max)):
    print((rms_max[i]/rms_max[-1])**2)
#%% rms of jh in y direction
sigmaH_list = [4,6,5,3.5,2.25]
sigmaP_list = [1,3,5,7,9]

rms_max = []

# files = glob.glob('*.txt')
files = glob.glob('1M\*.txt')
for i in range(len(files)):
    t1 = time.time()
    
    psi_list = np.loadtxt(files[i])
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N,sigmaP_list[i])
    jhx,jhy = hall(Ex,Ey,sigmaH_list[i])
    rms = np.sqrt(np.trapz((jhx**2+jhy**2),X*km))
    rms_max.append(np.max(rms))
    
    plt.figure('rms_jh_y')
    plt.plot(Y,rms,label='$\Sigma_{H}/\Sigma_{P}=$'+str(sigmaH_list[i]/sigmaP_list[i]))
    
    t2 = time.time()
    print('sigmaP = %sS done, time used:'%(sigmaP_list[i]),t2-t1)

plt.xlabel('$y(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.title('RMS magnitude of Hall current along y-axis')
plt.legend()
plt.grid()
plt.show()

for i in range(len(rms_max)):
    print(rms_max[i]/rms_max[-1])
  
print()

for i in range(len(rms_max)):
    print((rms_max[i]/rms_max[-1])**2)
#%% plot scaling relation
rms_max_plot = rms_max/rms_max[2]
ratio = np.array(sigmaH_list)/np.array(sigmaP_list)
B = np.linspace(min(ratio),max(ratio),1000)

def fit_func(x,a):
    return a*x

initial_guess = [1]
fit = curve_fit(fit_func,ratio,rms_max_plot,initial_guess,maxfev=1000000)
data_fit = fit_func(B,*fit[0])

plt.figure('rms_peaks',dpi=150)
plt.grid()
plt.plot(ratio,rms_max_plot,'o',label='Maximum')
plt.plot(B,data_fit,label='Fit = $\Sigma_{H}/\Sigma_{P}$')
plt.legend(prop={'size':12})
plt.xlabel('$\Sigma_{H}/\Sigma_{P}$')
plt.ylabel('Normalised magnitude')
plt.xticks(ratio,fontsize=9)
plt.title('Relation between RMS($\delta \mathbf{j}_{H}$) and conductance ratio')
plt.show()
print(fit[0][0])
#%%
fig, ax = plt.subplots()
ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=2)

ax.plot(range(1024))
plt.show()

