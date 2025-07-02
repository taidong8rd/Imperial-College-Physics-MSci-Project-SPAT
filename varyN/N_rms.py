# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:20:18 2023

@author: sz3119
"""

import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import time
from scipy.optimize import curve_fit
from BMfunctions import *

ky = (2*np.pi)/((2500/4)*km)

#%% create grid 
X = np.linspace(-2500,2500,1600) # km
Y = np.linspace(-2500,2500,1600)
xx,yy = np.meshgrid(X,Y)
#%% check
N = 5
plt.figure('FAC')
plt.plot(Y,delta_jr(Y*km,ky,N))
plt.xlabel('$y(km)$')
plt.ylabel('$\delta j_{r}$')
plt.xticks(np.arange(-2500,3000,1250))
plt.title('$\delta j_{r}$ at $z=z_{0}=25R_{E}$ as a function of y')
plt.grid()
plt.show()
#%% get potentials
N = [1,2,3,4,5]
for n in N:
    t1 = time.time()
    psi_list = potential(xx*km,yy*km,ky,n) 
    np.savetxt('psi_N=%s_1kx1k.txt'%(n),psi_list)
    t2 = time.time()
    print('N = %s done, time used:'%(n),t2-t1)
#%% rms of jp along x
N_list = np.array([1,2,3,4,5])
rms_max = []
# files = glob.glob('1M\*.txt')
files = glob.glob('2.56M\*.txt')
for i in range(len(files)):
    t1 = time.time()
    psi_list = np.loadtxt(files[i])
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N_list[i])
    jpx,jpy = pedersen(Ex,Ey)
    rms = np.sqrt(np.trapz((jpx**2+jpy**2).transpose(),Y*km))
    rms_max.append(np.max(rms))
    plt.figure('rms_jp_x',dpi=150)
    plt.plot(X,rms,label='N='+str(N_list[i]))
    # plt.plot(X,rms/np.sqrt(N_list[i]),label='N='+str(N_list[i]))
    t2 = time.time()
    print('N=%s done, time used:'%(N_list[i]),t2-t1)

plt.xlabel('$x(km)$')
# plt.xlabel('$k_{y}x$')
plt.xlim((-1000,1000))
plt.xticks(np.arange(-1000,1001,500))
plt.ylabel('RMS')
# plt.ylabel('RMS/$\sqrt{N}$')
plt.title('RMS of ionospheric currents along x-axis')
plt.legend()
plt.grid()
plt.show()

for i in range(len(rms_max)):
    print(rms_max[i]/rms_max[0])
  
print()

for i in range(len(rms_max)):
    print((rms_max[i]/rms_max[0])**2)
#%% plot x direction scaling relation 
rms_max_plot = rms_max/rms_max[0]  
A = np.linspace(1,5,1000)

def fit_func(x,a,b,r):
    return a*x**r+b

initial_guess = [1,1,1]
fit,cov = curve_fit(fit_func,N_list,rms_max_plot,initial_guess,maxfev=1000000)
data_fit = fit_func(A,*fit)

plt.figure('rms_jp_x_peaks',dpi=150)
plt.grid()
plt.plot(N_list,rms_max_plot,'o',label='Maximum')
# plt.plot(A,np.sqrt(A),label='Fit = $\sqrt{N}$')
plt.plot(A,data_fit,label='Fit = $aN^{%.2f}+b$'%(fit[-1]))
plt.legend()
plt.xlabel('N')
plt.ylabel('Normalised magnitude')
plt.title('Maximum of RMS of ionospheric currents along x-axis')
plt.xticks(np.linspace(1,5,5))
plt.show()
print(fit)
print(np.sqrt(np.diag(cov)))
#%% rms of jp along y
N_list = np.array([1,2,3,4,5])
rms_max = []
# files = glob.glob('1M\*.txt')
files = glob.glob('2.56M\*.txt')
for i in range(len(files)):
    t1 = time.time()
    psi_list = np.loadtxt(files[i])
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N_list[i])
    jpx,jpy = pedersen(Ex,Ey)
    rms = np.sqrt(np.trapz((jpx**2+jpy**2),X*km))
    rms_max.append(np.max(rms))
    plt.figure('rms_jp_y',dpi=150)
    plt.plot(Y,rms,label='N='+str(N_list[i]))
    t2 = time.time()
    print('N=%s done, time used:'%(N_list[i]),t2-t1)

plt.xlabel('$y(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.ylabel('RMS')
plt.title('RMS of ionospheric currents along y-axis')
plt.legend()
plt.grid()
plt.show()

for i in range(len(rms_max)):
    print(rms_max[i]/rms_max[0])
  
print()

for i in range(len(rms_max)):
    print((rms_max[i]/rms_max[0])**2)
#%% plot scaling relation
rms_max_plot = rms_max/rms_max[0] 
# rms_max_plot = np.array([0.961306,1.18863,1.35637,1.49942,1.62642])/0.961306 
A = np.linspace(1,5,1000)

def fit_func(x,a,b,r):
    return (a*x**r)+b

initial_guess = [0.1,0.9,-1.5]
fit,cov = curve_fit(fit_func,N_list,rms_max_plot,initial_guess,maxfev=1000000)
data_fit = fit_func(A,*fit)

plt.figure('rms_jp_y_peaks',dpi=150)
plt.grid()
plt.plot(N_list,rms_max_plot,'o',label='Maximum')
plt.plot(A,data_fit,label='Fit = $aN^{%.2f}+b$'%(fit[-1]))
plt.legend()
plt.xlabel('N')
plt.ylabel('Normalised magnitude')
plt.title('Maximum of RMS of ionospheric currents along y-axis')
plt.xticks(np.linspace(1,5,5))
plt.show()
print(fit)
print(np.sqrt(np.diag(cov)))
#%% rms of E in x direction ~ expect same as jp
N_list = np.array([1,2,3,4,5])
rms_max = []
files = glob.glob('1M\*.txt')
for i in range(len(files)):
    psi_list = np.loadtxt(files[i])
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N_list[i])
    rms = np.sqrt(np.trapz((Ex**2+Ey**2).transpose(),Y*km))
    rms_max.append(np.max(rms))
    plt.figure('rms_E_x')
    plt.plot(X,rms,label='N='+str(N_list[i]))

plt.xlabel('$x(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.title('RMS magnitude of electric field along x-axis')
plt.legend()
plt.grid()
plt.show()

for i in range(len(rms_max)):
    print(rms_max[i]/rms_max[0])
  
print()

for i in range(len(rms_max)):
    print((rms_max[i]/rms_max[0])**2)
#%% plot x direction scaling relation 
rms_max_plot = rms_max/rms_max[0]  
A = np.linspace(1,5,1000)

plt.figure('rms_E_x_peaks')
plt.grid()
plt.plot(N_list,rms_max_plot,'o',label='Maximum')
plt.plot(A,np.sqrt(A),label='Fit = $\sqrt{N}$')
plt.legend()
plt.xlabel('N')
plt.ylabel('Normalised magnitude')
plt.title('Relation between maximum of RMS(E) in x direction and N')
plt.xticks(np.linspace(1,5,5))
plt.show()
#%% rms of E along y ~ expect same as jp
N_list = np.array([1,2,3,4,5])
rms_max = []
files = glob.glob('1M\*.txt')
for i in range(len(files)):
    psi_list = np.loadtxt(files[i])
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N_list[i])
    rms = np.sqrt(np.trapz((Ex**2+Ey**2),X*km))
    rms_max.append(np.max(rms))
    plt.figure('rms_E_y')
    plt.plot(Y,rms,label='N='+str(N_list[i]))

plt.xlabel('$y(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.title('RMS magnitude of electric field along y-axis')
plt.legend()
plt.grid()
plt.show()

for i in range(len(rms_max)):
    print(rms_max[i]/rms_max[0])
  
print()

for i in range(len(rms_max)):
    print((rms_max[i]/rms_max[0])**2)
#%% plot scaling relation
rms_max_plot = rms_max/rms_max[0] 
# rms_max_plot = np.array([0.961306,1.18863,1.35637,1.49942,1.62642])/0.961306 
A = np.linspace(1,5,1000)

def fit_func(x,a,b,r):
    return (a/x**r)+b

initial_guess = [0.113,0.885,1.5]
fit = curve_fit(fit_func,N_list,rms_max_plot,initial_guess,maxfev=1000000)
data_fit = fit_func(A,*fit[0])

plt.figure('rms_E_y_peaks')
plt.grid()
plt.plot(N_list,rms_max_plot,'o',label='Maximum')
plt.plot(A,data_fit,label='Fit = $a/N^{%.2f}+b$'%(fit[0][-1]))
plt.legend()
plt.xlabel('N')
plt.ylabel('Normalised magnitude')
plt.title('Relation between maximum of RMS(E) in y direction and N')
plt.xticks(np.linspace(1,5,5))
plt.show()
print(fit[0])
#%% grids for B
Ym = np.linspace(-2500,2500,1000)
Z = np.linspace(0,25,1000) # RE
yym,zz = np.meshgrid(Ym,Z) # magnetopause gird, might differ in Y scale

X_ = np.linspace(-2250,2250,101)
Y_ = np.linspace(-2250,2250,101)
xx_,yy_ = np.meshgrid(X_,Y_)
plt.figure('grid for B')
plt.scatter(xx_,yy_)
#%% rms of ground Bhor in both directions
N_list = np.array([1,2,3,4,5])
h = 110*km

rmsx_max = []
rmsy_max = []
files = glob.glob('1M\*.txt')
for i in range(len(files)):
    t1 = time.time()
    
    psi_list = np.loadtxt(files[i])
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N_list[i])
    
    jpx,jpy = pedersen(Ex,Ey)
    jhx,jhy = hall(Ex,Ey)
    jmy,jmz = delta_J_im(yym*km,zz*RE,ky,N_list[i])
    
    Bpx,Bpy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jpx,jpy,h)
    Bhx,Bhy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jhx,jhy,h)
    Bmx,Bmy = magnetic_field_m(xx_*km,yy_*km,yym*km,zz*RE,jmy,jmz,h)

    Bx = Bpx + Bhx + Bmx
    By = Bpy + Bhy + Bmy
    
    rmsx = np.sqrt(np.trapz((Bx**2+By**2).transpose(),Y_*km))
    rmsx_max.append(np.max(rmsx))
    
    rmsy = np.sqrt(np.trapz((Bx**2+By**2),X_*km))
    rmsy_max.append(np.max(rmsy))
    
    plt.figure('rms_Bhor(0km)_x')
    plt.plot(X_,rmsx,label='N='+str(N_list[i]))
    
    plt.figure('rms_Bhor(0km)_y')
    plt.plot(Y_,rmsy,label='N='+str(N_list[i]))
    
    t2 = time.time()
    print('N = %s done, time used:'%(N_list[i]),t2-t1)

plt.figure('rms_Bhor(0km)_x')
plt.xlabel('$x(km)$')
# plt.xticks(np.arange(-2500,2501,500))
plt.ylabel('RMS')
plt.title('RMS magnitude of ground $\delta \mathbf{B}_{hor}$ along x-axis')
plt.legend()
plt.grid()
plt.savefig('rms_Bhor(0km)_x')

plt.figure('rms_Bhor(0km)_y')
plt.xlabel('$y(km)$')
# plt.xticks(np.arange(-2500,2501,500))
plt.ylabel('RMS')
plt.title('RMS magnitude of ground $\delta \mathbf{B}_{hor}$ along y-axis')
plt.legend()
plt.grid()
plt.savefig('rms_Bhor(0km)_y')

np.savetxt('N_rmsx_B.txt',rmsx_max)
np.savetxt('N_rmsy_B.txt',rmsy_max)

# for i in range(len(rms_max)):
#     print(rms_max[i]/rms_max[0])
  
# print()

# for i in range(len(rms_max)):
#     print((rms_max[i]/rms_max[0])**2)
    
# print()

# for i in range(len(rms_max)-1):
#     print((rms_max[i+1]/rms_max[i]))
#%% plot x direction scaling relation
rmsx_max_plot = rmsx_max/rmsx_max[0]  
A = np.linspace(1,5,1000)

def fit_func(x,a,r,b):
    return (a*x**r)+b

initial_guess = [1,1,1]
fit = curve_fit(fit_func,N_list,rmsx_max_plot,initial_guess,maxfev=1000000)
data_fit = fit_func(A,*fit[0])

plt.figure('rms_Bhor(0km)_x_peaks')
plt.grid()
plt.plot(N_list,rmsx_max_plot,'o',label='Maximum')
plt.plot(A,data_fit,label='Fit = $%.2f \cdot N^{%.2f}+%.2f$'%(tuple(fit[0])))
plt.legend()
plt.xlabel('N')
plt.ylabel('Normalised magnitude')
plt.title('Maximum of RMS($\delta \mathbf{B}_{hor}$) in x direction v.s. N')
plt.xticks(np.linspace(1,5,5))
plt.savefig('rms_Bhor(0km)_x_peaks')
plt.show()
print(fit[0])
#%% plot y direction scaling relation
rmsy_max_plot = rmsy_max/rmsy_max[0]  
A = np.linspace(1,5,1000)

# def fit_func(x,a,b,r):
#     return (a*x**r)+b

def fit_func(x,a,r,b):
    return a*np.exp(-x**r)+b

initial_guess = [1,1,1]
fit = curve_fit(fit_func,N_list,rmsy_max_plot,initial_guess,maxfev=1000000)
data_fit = fit_func(A,*fit[0])

plt.figure('rms_Bhor(0km)_y_peaks')
plt.grid()
plt.plot(N_list,rmsy_max_plot,'o',label='Maximum')
plt.plot(A,data_fit,label='Fit = $%.2f \cdot exp(-N^{%.2f})+%.2f$'%(tuple(fit[0])))
plt.legend()
plt.xlabel('N')
plt.ylabel('Normalised magnitude')
plt.title('Maximum of RMS($\delta \mathbf{B}_{hor}$) in y direction v.s. N')
plt.xticks(np.linspace(1,5,5))
plt.savefig('rms_Bhor(0km)_y_peaks')
plt.show()
print(fit[0])
print(np.sqrt(fit[1]))
