# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:28:36 2023

@author: zhang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import time
from BMfunctions import *

ky = (2*np.pi)/(2500*km)
#%% create grids
X = np.linspace(-2500,2500,1600) # km
Y = np.linspace(-2500,2500,1600)
xx,yy = np.meshgrid(X,Y) # ionosphere grid

Ym = np.linspace(-2500,2500,1600)
Z = np.linspace(0,25,1600) # RE
yym,zz = np.meshgrid(Ym,Z) # magnetopause gird
#%%
t1 = time.time()
# psi_list = potential(xx*km,yy*km,ky)
# np.savetxt('psi_stdkm_500x500.txt',psi_list)
# psi_list = np.loadtxt('psi_stdkm_1kx1k.txt')
psi_list = np.loadtxt('psi_stdkm_1.6kx1.6k.txt')
Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky)
t2 = time.time()
print('time used:',t2-t1)
#%% calculate currents
t1 = time.time()
jpx,jpy = pedersen(Ex,Ey)
jhx,jhy = hall(Ex,Ey)
jmy,jmz = delta_J_im(yym*km,zz*RE,ky)
t2 = time.time()
print('time used:',t2-t1)
#%% create grid where I want the magnetic fields
# X_ = np.linspace(200,2250,8)
# Y_ = np.linspace(-2250,2250,7)

X_ = np.linspace(-2000,2000,10)
Y_ = np.linspace(-2000,2000,10)

xx_,yy_ = np.meshgrid(X_,Y_)
plt.figure('grid for B')
plt.scatter(xx_,yy_)
#%% calculate all contributions to B
h = 110*km

Bpx,Bpy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jpx,jpy,h)
Bhx,Bhy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jhx,jhy,h)
Bmx,Bmy = magnetic_field_m(xx_*km,yy_*km,yym*km,zz*RE,jmy,jmz,h)

Bx = Bpx + Bhx + Bmx
By = Bpy + Bhy + Bmy
#%% rms
def double_int(Bx,By,X,Y):
    temp = np.trapz(np.trapz((Bx**2+By**2),X),Y)
    return np.sqrt(temp)

print(double_int(Bpx,Bpy,X_*km,Y_*km))
print(double_int(Bhx,Bhy,X_*km,Y_*km))
print(double_int(Bmx,Bmy,X_*km,Y_*km))




