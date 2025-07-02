# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 00:24:17 2023

@author: zhang
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

# X_ = np.linspace(200,2250,8)
# Y_ = np.linspace(-2250,2250,7)
X_ = np.linspace(200,2000,8)
Y_ = np.linspace(-2000,2000,7)
xx_,yy_ = np.meshgrid(X_,Y_) # grid pts where I want magnetic fields
plt.figure('grid for B')
plt.scatter(xx_,yy_)
#%% plot FAC
plt.figure('FAC')
plt.plot(Y,delta_jr(Y*km,ky,N))
plt.xlabel('$y(km)$')
plt.ylabel('$\delta j_{r}$')
plt.xticks(np.arange(-2500,3000,1250))
plt.title('$\delta j_{r}$ at $z=z_{0}=25R_{E}$ as a function of y')
plt.grid()
plt.show()
#%% explore how a B contribution changes with sigmaP
h = 110*km
sigmaP_list = [1,3,5,7,9]

color_list = ['silver','-','grey','-','black']
# color_list = ['lime','-','limegreen','-','green']
#color_list = ['violet','-','fuchsia','-','purple']
# color_list = ['deepskyblue','-','dodgerblue','-','royalblue']
# files = glob.glob('*.txt')
files = glob.glob('1M\*.txt')

for i in range(len(files)):
    if 2*i <= 4:
        t1 = time.time()
        
        psi_list = np.loadtxt(files[2*i])
        Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N,sigmaP_list[2*i])
        jpx,jpy = pedersen(Ex,Ey,sigmaP_list[2*i])
        jhx,jhy = hall(Ex,Ey,5)
        jmy,jmz = delta_J_im(yym*km,zz*RE,ky,N)
        
        Bpx,Bpy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jpx,jpy,h)
        Bhx,Bhy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jhx,jhy,h)
        Bmx,Bmy = magnetic_field_m(xx_*km,yy_*km,yym*km,zz*RE,jmy,jmz,h)
        
        Bx = Bpx + Bhx + Bmx
        By = Bpy + Bhy + Bmy
       
        Bpx_ = Bpx/np.sqrt(Bx**2+By**2) # normalised by total field at each position
        Bpy_ = Bpy/np.sqrt(Bx**2+By**2)
        Bhx_ = Bhx/np.sqrt(Bx**2+By**2)
        Bhy_ = Bhy/np.sqrt(Bx**2+By**2)
        Bmx_ = Bmx/np.sqrt(Bx**2+By**2)
        Bmy_ = Bmy/np.sqrt(Bx**2+By**2)
        Bx_ = Bx/np.sqrt(Bx**2+By**2)
        By_ = By/np.sqrt(Bx**2+By**2)
        
        plt.figure('B_ground')
#        plt.quiver(xx_,yy_,Bpx_,-Bpy_,label='$\Sigma_{P}=$'+str(sigmaP_list[2*i]),color=color_list[2*i])
        # plt.quiver(xx_,yy_,Bhx_,-Bhy_,label='$\Sigma_{P}=$'+str(sigmaP_list[2*i]),color=color_list[2*i])
        # plt.quiver(xx_,yy_,Bmx_,-Bmy_,label='$\Sigma_{P}=$'+str(sigmaP_list[2*i]),color=color_list[2*i])
        plt.quiver(xx_,yy_,Bx_,-By_,label='$\Sigma_{P}=$'+str(sigmaP_list[2*i]),color=color_list[2*i])
        
        t2 = time.time()
        print('sigmaP = %sS done, time used:'%(sigmaP_list[2*i]),t2-t1)

plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
# plt.xticks(np.arange(0,2501,500))
# plt.yticks(np.arange(-3000,3001,500))
# plt.xlim(-100,3000)
# plt.ylim(2750,-2750)
plt.xticks(np.arange(0,2501,500))
plt.yticks(np.arange(-2500,2501,500))
plt.xlim(-100,2600)
plt.ylim(2500,-2500)
plt.legend(loc=1)
plt.title('Ground $\delta \mathbf{B}_{hor}$ at different $\Sigma_{P}$')
plt.show()
#%% plot all contributions to B like before
h = 110*km
sigmaP = 5

# psi_list = np.loadtxt('psi_5S_1.6kx1.6k.txt')
psi_list = np.loadtxt('1M\psi_%sS_1kx1k.txt'%(sigmaP))
Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N,sigmaP)
jpx,jpy = pedersen(Ex,Ey,sigmaP)
jhx,jhy = hall(Ex,Ey,5)
jmy,jmz = delta_J_im(yym*km,zz*RE,ky,N)

Bpx,Bpy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jpx,jpy,h)
Bhx,Bhy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jhx,jhy,h)
Bmx,Bmy = magnetic_field_m(xx_*km,yy_*km,yym*km,zz*RE,jmy,jmz,h)

Bx = Bpx + Bhx + Bmx
By = Bpy + Bhy + Bmy

Bpx_ = Bpx/np.sqrt(Bx**2+By**2) # normalised by total field at each position
Bpy_ = Bpy/np.sqrt(Bx**2+By**2)
Bhx_ = Bhx/np.sqrt(Bx**2+By**2)
Bhy_ = Bhy/np.sqrt(Bx**2+By**2)
Bmx_ = Bmx/np.sqrt(Bx**2+By**2)
Bmy_ = Bmy/np.sqrt(Bx**2+By**2)
Bx_ = Bx/np.sqrt(Bx**2+By**2)
By_ = By/np.sqrt(Bx**2+By**2)

plt.figure('B_contributions')
plt.quiver(xx_,yy_,Bpx_,-Bpy_,label='$\delta B_{P}$',color='fuchsia')
plt.quiver(xx_,yy_,Bhx_,-Bhy_,label='$\delta B_{H}$',color='limegreen')
plt.quiver(xx_,yy_,Bmx_,-Bmy_,label='$\delta B_{MP}$',color='deepskyblue')
plt.quiver(xx_,yy_,Bx_,-By_,label='$\delta B_{hor}$',color='black')

plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(0,2501,500))
plt.yticks(np.arange(-3000,3001,500))
plt.xlim(-100,3000)
plt.ylim(2750,-2750)
plt.legend(loc=1)
plt.title('Contributions to ground $\delta B_{hor}$ with $\Sigma_{P}=$'+str(sigmaP))
plt.show()