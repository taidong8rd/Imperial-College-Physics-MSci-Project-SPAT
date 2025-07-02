# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:39:15 2023

@author: zhang
"""

import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import time
from scipy.optimize import curve_fit
from BMfunctions import *
from matplotlib.colors import CenteredNorm

#%% create grids
X = np.linspace(-2500,2500,1000)
Y = np.linspace(-2500,2500,1000)
xx,yy = np.meshgrid(X,Y) # ionosphere grid
#%% get potentials
A = np.array([0,1,2,3,4])
ky_list = (2*np.pi)/(625*km*(np.sqrt(2))**(A))
for i in range(len(ky_list)):
    t1 = time.time()
    psi_list = potential(xx*km,yy*km,ky_list[i],N) 
    np.savetxt('1M\psi_ky%s_1kx1k.txt'%(A[i]),psi_list)
    t2 = time.time()
    print('ky No.%s done, time used:'%(A[i]),t2-t1)
#%% rms of jp along x
A = np.array([0,1,2,3,4])
ky_list = (2*np.pi)/(625*km*(np.sqrt(2))**(A))

rms_max = []
# files = glob.glob('*.txt')
files = glob.glob('1M\*.txt')

for i in range(len(files)):
    t1 = time.time()
    
    psi_list = np.loadtxt(files[i])
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky_list[i])
    jpx,jpy = pedersen(Ex,Ey)
    rms = np.sqrt(np.trapz((jpx**2+jpy**2).transpose(),Y*km))
    rms_max.append(np.max(rms))
    
    plt.figure('rms_jp_x')
#    plt.plot(X,rms,label='ky'+str(A[i]))
    plt.plot(ky_list[i]*X,rms/np.sqrt(ky_list[i]),label='$k_{y%s}$'%str(A[i]))
    
    t2 = time.time()
    print('ky No.%s done, time used:'%(A[i]),t2-t1)

#plt.xlabel('$x(km)$')
plt.xlabel('$k_{y}x$')
#plt.xticks(np.arange(-2500,2501,500))
plt.ylabel('RMS/$\sqrt{k_{y}}$')
plt.title('RMS magnitude of Pedersen current along x-axis')
plt.legend()
plt.grid()
plt.show()

for i in range(len(rms_max)):
    print(rms_max[i]/rms_max[-1])
  
print()

for i in range(len(rms_max)):
    print((rms_max[i]/rms_max[-1])**2)
#%% rms of jp along y
A = np.array([0,1,2,3,4])
ky_list = (2*np.pi)/(625*km*(np.sqrt(2))**(A))

rms_max = []
# files = glob.glob('*.txt')
files = glob.glob('1M\*.txt')

for i in range(len(files)):
    t1 = time.time()
    
    psi_list = np.loadtxt(files[i])
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky_list[i])
    jpx,jpy = pedersen(Ex,Ey)
    rms = np.sqrt(np.trapz((jpx**2+jpy**2),X*km))
    rms_max.append(np.max(rms))
    
    plt.figure('Ky_rms_jp_y',dpi=150)
    # plt.plot(Y,rms,label='$k_{y%s}$'%str(A[i]))
    plt.plot(ky_list[i]*Y,rms/np.sqrt(ky_list[i]),label='$k_{y%s}$'%str(A[i]))
    
    t2 = time.time()
    print('ky No.%s done, time used:'%(A[i]),t2-t1)

# plt.xlabel('$y(km)$',size=14)
plt.xlabel('$k_{y}y$',size=14)
# plt.xticks(np.arange(-2500,2501,500))
# plt.ylabel('RMS',size=14)
plt.ylabel('RMS/$\sqrt{k_{y}}$',size=14)
plt.title('RMS of ionospheric currents along y-axis')
plt.legend(prop={'size': 12})
plt.grid()
plt.show()

for i in range(len(rms_max)):
    print(rms_max[i]/rms_max[0])
  
print()

for i in range(len(rms_max)):
    print((rms_max[i]/rms_max[0])**2)
#%% plot scaling relation
rms_max_plot = rms_max/rms_max[-1]  
B = np.linspace(min(ky_list),max(ky_list),1000)

def fit_func(x,a):
    return a*np.sqrt(x)

initial_guess = [625]
fit = curve_fit(fit_func,ky_list,rms_max_plot,initial_guess,maxfev=1000000)
data_fit = fit_func(B,*fit[0])

plt.figure('rms_jp_y_peaks')
plt.grid()
plt.plot(ky_list,rms_max_plot,'o',label='Maximum')
plt.plot(B,data_fit,label='Fit = $%.1f\sqrt{k_{y}}$'%fit[0][0])
# plt.plot(B,np.sqrt(B)*625,label='$r_{N}=625\sqrt{ky_{N}}$')
plt.legend()
plt.xlabel('ky')
plt.ylabel('Normalised magnitude')
plt.title('Relation between maximum of RMS($\delta j_{p}$) and $k_{y}$')
plt.xticks(ky_list,['$k_{y0}$','$k_{y1}$','$k_{y2}$','$k_{y3}$','$k_{y4}$'])
# plt.yscale('log')
plt.show()
print(fit[0][0])
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
A = np.array([0,1,2,3,4])
ky_list = (2*np.pi)/(625*km*(np.sqrt(2))**(A))
h = 110*km

rmsx_max = []
rmsy_max = []
files = glob.glob('1M\*.txt')
for i in range(len(files)):
    t1 = time.time()
    
    psi_list = np.loadtxt(files[i])
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky_list[i])
    
    jpx,jpy = pedersen(Ex,Ey)
    jhx,jhy = hall(Ex,Ey)
    jmy,jmz = delta_J_im(yym*km,zz*RE,ky_list[i])
    
    Bpx,Bpy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jpx,jpy,h)
    Bhx,Bhy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jhx,jhy,h)
    Bmx,Bmy = magnetic_field_m(xx_*km,yy_*km,yym*km,zz*RE,jmy,jmz,h)

    Bx = Bpx + Bhx + Bmx
    By = Bpy + Bhy + Bmy
    
    rmsx = np.sqrt(np.trapz((Bx**2+By**2).transpose(),Y_*km))
    rmsx_max.append(np.max(rmsx))
    
    rmsy = np.sqrt(np.trapz((Bx**2+By**2),X_*km))
    rmsy_max.append(np.max(rmsy))
    
    plt.figure('rms_Bhor(0km)_x',dpi=150)
    plt.plot(X_,rmsx,label='$k_{y%s}$'%str(A[i]))
    
    plt.figure('rms_Bhor(0km)_y',dpi=150)
    plt.plot(Y_,rmsy,label='$k_{y%s}$'%str(A[i]))
    
    t2 = time.time()
    print('ky%s done, time used:'%(A[i]),t2-t1)

plt.figure('rms_Bhor(0km)_x')
plt.xlabel('$x(km)$')
# plt.xticks(np.arange(-2500,2501,500))
plt.ylabel('RMS')
plt.title('RMS magnitude of ground $\delta \mathbf{B}_{hor}$ along x-axis')
plt.legend(prop={'size': 12})
plt.grid()
plt.savefig('rms_Bhor(0km)_x')

plt.figure('rms_Bhor(0km)_y')
plt.xlabel('$y(km)$')
# plt.xticks(np.arange(-2500,2501,500))
plt.ylabel('RMS')
plt.title('RMS magnitude of ground $\delta \mathbf{B}_{hor}$ along y-axis')
plt.legend(prop={'size': 12})
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
B = np.linspace(min(ky_list),max(ky_list),1000)

def fit_func(x,a,r,b):
    return (a*x**r)+b

initial_guess = [1,1,1]
fit = curve_fit(fit_func,ky_list,rmsx_max_plot,initial_guess,maxfev=1000000)
data_fit = fit_func(B,*fit[0])

plt.figure('rms_Bhor(0km)_x_peaks',dpi=150)
plt.grid()
plt.plot(ky_list,rmsx_max_plot,'o',label='Maximum')
plt.plot(B,data_fit,label='Fit = $%.2f \cdot (k_y)^{%.2f}+%.2f$'%(tuple(fit[0])))
plt.xlabel('$k_y$')
plt.xticks(ky_list,['$k_{y0}$','$k_{y1}$','$k_{y2}$','$k_{y3}$','$k_{y4}$'])
plt.ylabel('Normalised magnitude')
plt.legend()
plt.title('Maximum of RMS($\delta \mathbf{B}_{hor}$) in x direction v.s. $k_y$')
plt.savefig('rms_Bhor(0km)_x_peaks')
plt.show()
print(fit[0])
#%% plot y direction scaling relation
rmsy_max_plot = rmsy_max/rmsy_max[0]  
B = np.linspace(min(ky_list),max(ky_list),1000)

def fit_func(x,a,r,b):
    return (a*x**r)+b

# def fit_func(x,a,r,b):
#     return a*np.exp(-x**r)+b

initial_guess = [1,1,1]
fit = curve_fit(fit_func,ky_list,rmsy_max_plot,initial_guess,maxfev=1000000)
data_fit = fit_func(B,*fit[0])

plt.figure('rms_Bhor(0km)_y_peaks',dpi=150)
plt.grid()
plt.plot(ky_list,rmsy_max_plot,'o',label='Maximum')
plt.plot(B,data_fit,label='Fit = $%.2f \cdot (k_y)^{%.2f}+%.2f$'%(tuple(fit[0])))
plt.xlabel('$k_y$')
plt.xticks(ky_list,['$k_{y0}$','$k_{y1}$','$k_{y2}$','$k_{y3}$','$k_{y4}$'])
plt.ylabel('Normalised magnitude')
plt.legend()
plt.title('Maximum of RMS($\delta \mathbf{B}_{hor}$) in y direction v.s. $k_y$')
plt.savefig('rms_Bhor(0km)_y_peaks')
plt.show()
print(fit[0])
print(np.sqrt(fit[1]))
