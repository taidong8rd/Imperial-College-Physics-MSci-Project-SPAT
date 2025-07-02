# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 21:55:56 2023

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

Ym = np.linspace(-2500,2500,1000)
Z = np.linspace(0,25,1000) # RE
yym,zz = np.meshgrid(Ym,Z) # magnetopause gird
#%% plot FAC
A = 4
ky = (2*np.pi)/(625*km*(np.sqrt(2))**(A))

plt.figure('FAC')
plt.plot(Y,delta_jr(Y*km,ky))
plt.xlabel('$y(km)$')
plt.ylabel('$\delta j_{r}$')
plt.xticks(np.arange(-2500,3000,1250))
plt.title('$\delta j_{r}$ at $z=z_{0}=25R_{E}$ as a function of y')
plt.grid()
plt.show()
#%% plot magnetopause current
A = 2
ky = (2*np.pi)/(625*km*(np.sqrt(2))**(A))

jmy,jmz = delta_J_im(yym*km,zz*RE,ky)

plt.figure('magnetopause current')
a = None
b = None
c = 40
_yym = yym[a:b:c,a:b:c]
_zz = zz[a:b:c,a:b:c]
_jmy = jmy[a:b:c,a:b:c]
_jmz = jmz[a:b:c,a:b:c]
plt.quiver(_yym,_zz,_jmy,_jmz)
# plt.scatter(_yym,_zz)
plt.xlabel('$y(km)$')
plt.ylabel('$z(R_{E})$')
plt.xticks(np.arange(-2500,3000,500))
plt.yticks(np.arange(0,30,5))
plt.title('Magnetopause current ($\delta \mathbf{J}$)')
plt.show()
#%% plot potential
A = 4
ky = (2*np.pi)/(625*km*(np.sqrt(2))**(A))

# psi_list = np.loadtxt('2.56M\psi_ky%s_1.6kx1.6k.txt'%(A))
psi_list = np.loadtxt('1M\psi_ky%s_1kx1k.txt'%(A))
# psi_list = potential(xx*km,yy*km) 
Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky)
jpx,jpy = pedersen(Ex,Ey)
jhx,jhy = hall(Ex,Ey)

plt.figure('potential',figsize=(9.6,7.2))
plt.contourf(xx,yy,psi_list,300)
ticks = np.linspace(np.max(psi_list),np.min(psi_list),7)
ticks = np.round(ticks,11)
plt.colorbar(ticks=ticks)
plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.yticks(np.arange(-2500,2501,500))
plt.xlim(-2500,2500)
plt.ylim(2500,-2500)
plt.title('Electric potential at N-ionosphere ($z_{0}=25R_{E}$)')
plt.show()
#%% plot pedersen current
plt.figure('Pedersen current')
# plt.quiver(xx,yy,jpx,jpy)
a = 1
b = -a
c = 1
_xx = xx[a:b:c,a:b:c]
_yy = yy[a:b:c,a:b:c]
_jpx = jpx[a:b:c,a:b:c]
_jpy = jpy[a:b:c,a:b:c]
modulus = np.sqrt(_jpx**2+_jpy**2)/np.max(np.sqrt(_jpx**2+_jpy**2))
plt.streamplot(_xx,_yy,_jpx,_jpy,color=modulus)
# plt.streamplot(xx,yy,jpx,jpy,color=np.sqrt(jpx**2+jpy**2)/np.max(np.sqrt(jpx**2+jpy**2)))
plt.colorbar()
# plt.quiver(_xx,_yy,_jpx,-_jpy)
plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.yticks(np.arange(-2500,2501,500))
plt.xlim(-2500,2500)
plt.ylim(2500,-2500)
plt.title('Pedersen Current at N-ionosphere ($z_{0}=25R_{E}$)')
plt.show()
#%% plot Hall current
plt.figure('Hall current')
a = 1
b = -a
c = 1
_xx = xx[a:b:c,a:b:c]
_yy = yy[a:b:c,a:b:c]
_jhx = jhx[a:b:c,a:b:c]
_jhy = jhy[a:b:c,a:b:c]
modulus = np.sqrt(_jhx**2+_jhy**2)/np.max(np.sqrt(_jhx**2+_jhy**2))
plt.streamplot(_xx,_yy,_jhx,_jhy,color=modulus)
# plt.streamplot(xx,yy,jhx,jhy,color=np.sqrt(jhx**2+jhy**2)/np.max(np.sqrt(jhx**2+jhy**2)))
plt.colorbar()
plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.yticks(np.arange(-2500,2501,500))
plt.xlim(-2500,2500)
plt.ylim(2500,-2500)
plt.title('Hall Current at N-ionosphere ($z_{0}=25R_{E}$)')
plt.show()
#%% grid pts where I want magnetic fields
X_ = np.linspace(200,2250,8)
Y_ = np.linspace(-2250,2250,7)
xx_,yy_ = np.meshgrid(X_,Y_) 
plt.figure('grid for B')
plt.scatter(xx_,yy_)
#%% explore how a B contribution changes with ky
h = 110*km

A = np.array([0,1,2,3,4])
ky_list = (2*np.pi)/(625*km*(np.sqrt(2))**(A))

color_list = ['silver','-','grey','-','black']
# color_list = ['lime','-','limegreen','-','green']
# color_list = ['violet','-','fuchsia','-','purple']
# color_list = ['deepskyblue','-','dodgerblue','-','royalblue']
# files = glob.glob('2.56M\*.txt')
files = glob.glob('1M\*.txt')

for i in range(len(files)):
    if 2*i <= 4:
        t1 = time.time()
        
        psi_list = np.loadtxt(files[2*i])
        Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky_list[2*i])
        jpx,jpy = pedersen(Ex,Ey)
        jhx,jhy = hall(Ex,Ey)
        jmy,jmz = delta_J_im(yym*km,zz*RE,ky_list[2*i])
        
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
        
        plt.figure('B_ground',dpi=150)
        # plt.quiver(xx_,yy_,Bpx_,-Bpy_,label='ky'+str(A[2*i]),color=color_list[2*i])
        # plt.quiver(xx_,yy_,Bhx_,-Bhy_,label='ky'+str(A[2*i]),color=color_list[2*i])
        # plt.quiver(xx_,yy_,Bmx_,-Bmy_,label='ky'+str(A[2*i]),color=color_list[2*i])
        plt.quiver(xx_,yy_,Bx_,-By_,label='$k_{y%s}$'%str(A[2*i]),color=color_list[2*i])
        
        t2 = time.time()
        print('ky No.%s done, time used:'%(A[2*i]),t2-t1)

plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(0,2501,500))
plt.yticks(np.arange(-3000,3001,500))
plt.xlim(-100,2800)
plt.ylim(2750,-2750)
plt.legend(loc=1)
plt.title('Ground $\delta \mathbf{B}_{hor}$ at different $k_y$')
plt.show()
#%% plot all contributions to B like before
h = 110*km

A = 4
ky = (2*np.pi)/(625*km*(np.sqrt(2))**(A))

#psi_list = np.loadtxt('2.56M\psi_ky%s_1.6kx1.6k.txt'%(A))
psi_list = np.loadtxt('1M\psi_ky%s_1kx1k.txt'%(A))
Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky)
jpx,jpy = pedersen(Ex,Ey)
jhx,jhy = hall(Ex,Ey)
jmy,jmz = delta_J_im(yym*km,zz*RE,ky)

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
plt.title('Contributions to ground $\delta B_{hor}$ with $k_{y}=k_{y%s}$'%(A))
plt.show()