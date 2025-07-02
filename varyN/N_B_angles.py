# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:55:44 2023

@author: sz3119
"""

import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import time
from scipy.optimize import curve_fit
from BMfunctions import *
from matplotlib.colors import CenteredNorm

ky = (2*np.pi)/((2500/4)*km)

#%% create grids
X = np.linspace(-2500,2500,1000)
Y = np.linspace(-2500,2500,1000)
xx,yy = np.meshgrid(X,Y) # ionosphere grid

Ym = np.linspace(-2500,2500,1000)
Z = np.linspace(0,25,1000) # RE
yym,zz = np.meshgrid(Ym,Z) # magnetopause gird

X_ = np.linspace(-1000,1000,101)
Y_ = np.linspace(-2000,2000,101)
xx_,yy_ = np.meshgrid(X_,Y_) # grid pts where I want magnetic fields
plt.figure('grid for B')
plt.scatter(xx_,yy_)
#%% calculate Bhor(ground) and Bmp(400km) only
N = 2

# psi_list = np.loadtxt('psi_1S_1.6kx1.6k.txt')
psi_list = np.loadtxt('1M\psi_N=%s_1kx1k.txt'%(N))
Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N)
jpx,jpy = pedersen(Ex,Ey)
jhx,jhy = hall(Ex,Ey)
jmy,jmz = delta_J_im(yym*km,zz*RE,ky,N)

# ground, need Bhor only
Bpx,Bpy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jpx,jpy,110*km)
Bhx,Bhy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jhx,jhy,110*km)
Bmx_g,Bmy_g = magnetic_field_m(xx_*km,yy_*km,yym*km,zz*RE,jmy,jmz,110*km)

Bx = Bpx + Bhx + Bmx_g
By = Bpy + Bhy + Bmy_g

# above ionosphere, need Bmp only
Bmx,Bmy = magnetic_field_m(xx_*km,yy_*km,yym*km,zz*RE,jmy,jmz,110*km-400*km)

# all normalise to ground Bhor
Bmx_ = Bmx/np.sqrt(Bx**2+By**2)
Bmy_ = Bmy/np.sqrt(Bx**2+By**2)
Bx_ = Bx/np.sqrt(Bx**2+By**2)
By_ = By/np.sqrt(Bx**2+By**2)
#%% quiver-plot them
# a = 1
# b = -a
# c = 1
# _xx = xx_[a:b:c,a:b:c]
# _yy = yy_[a:b:c,a:b:c]
# _Bx = Bx_[a:b:c,a:b:c]
# _By = By_[a:b:c,a:b:c]
# _Bmx = Bmx_[a:b:c,a:b:c]
# _Bmy = Bmy_[a:b:c,a:b:c]

plt.figure('B_altitudes',dpi=150)
# plt.quiver(_xx,_yy,_Bx,-_By,label='$\delta B_{hor}(0km)$',color='black')
# plt.quiver(_xx,_yy,_Bmx,-_Bmy,label='$\delta B_{MP}(400km)$',color='dodgerblue')
plt.quiver(xx_,yy_,Bx_,-By_,label='$\delta B_{hor}(0km)$',color='black')
plt.quiver(xx_,yy_,Bmx_,-Bmy_,label='$\delta B_{MP}(400km)$',color='dodgerblue')
    
plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
# plt.xticks(np.arange(0,2501,500))
# plt.yticks(np.arange(-3000,3001,500))
# plt.ylim(2750,-2750)
# plt.xlim(-100,3150)
plt.legend(loc=2,prop={'size':8})
plt.gca().invert_yaxis()
plt.title('Magnetic fields at different altitudes with N=%s'%(N))
plt.show()
#%% work out angle between them
dot_product = Bx*Bmx + By*Bmy
B_modulus = np.sqrt(Bx**2+By**2)
Bm_modulus = np.sqrt(Bmx**2+Bmy**2)

cos_theta = dot_product/(B_modulus*Bm_modulus)
theta = np.arccos(cos_theta)*(180/np.pi)
print(theta)
#%% color plot for the angles
name = 'angles_highres_OCB_N=%s'%(N)
plt.figure(name,dpi=150)
plt.contourf(xx_,yy_,theta,norm=CenteredNorm(90),levels=1000,cmap='seismic')
ticks = np.array([np.min(theta),(np.min(theta)+90)/2,90,(np.max(theta)+90)/2,np.max(theta)])
plt.colorbar(ticks=ticks,format='%.1f',label='Degree($\degree$)')
plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.gca().invert_yaxis()
plt.title('Rotation angles (N=%s)'%(N))
plt.savefig(name)
plt.show()
#%% do more at a time
N_list = [1,2,3,4,5]

for N in N_list:
    psi_list = np.loadtxt('1M\psi_N=%s_1kx1k.txt'%(N))
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N)
    jpx,jpy = pedersen(Ex,Ey)
    jhx,jhy = hall(Ex,Ey)
    jmy,jmz = delta_J_im(yym*km,zz*RE,ky,N)
    
    # ground, need Bhor only
    Bpx,Bpy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jpx,jpy,110*km)
    Bhx,Bhy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jhx,jhy,110*km)
    Bmx_g,Bmy_g = magnetic_field_m(xx_*km,yy_*km,yym*km,zz*RE,jmy,jmz,110*km)
    
    Bx = Bpx + Bhx + Bmx_g
    By = Bpy + Bhy + Bmy_g
    
    # above ionosphere, need Bmp only
    Bmx,Bmy = magnetic_field_m(xx_*km,yy_*km,yym*km,zz*RE,jmy,jmz,110*km-400*km)
    
    dot_product = Bx*Bmx + By*Bmy
    B_modulus = np.sqrt(Bx**2+By**2)
    Bm_modulus = np.sqrt(Bmx**2+Bmy**2)
    
    cos_theta = dot_product/(B_modulus*Bm_modulus)
    theta = np.arccos(cos_theta)*(180/np.pi)
 
    name = 'angles_highres_OCB_N=%s'%(N)
    plt.figure(name,dpi=150)
    plt.contourf(xx_,yy_,theta,norm=CenteredNorm(90),levels=1000,cmap='seismic')
    ticks = np.array([np.min(theta),(np.min(theta)+90)/2,90,(np.max(theta)+90)/2,np.max(theta)])
    plt.colorbar(ticks=ticks,format='%.1f',label='Degree($\degree$)')
    plt.xlabel('$x(km)$')
    plt.ylabel('$y(km)$')
    plt.gca().invert_yaxis()
    plt.title('Rotation angles (N=%s)'%(N))
    plt.savefig(name)
    # plt.show()
#%% test single point
x = 230*km
y = 1000*km

Bpx,Bpy = magnetic_single_ph(x,y,xx*km,yy*km,jpx,jpy,110*km)
Bhx,Bhy = magnetic_single_ph(x,y,xx*km,yy*km,jhx,jhy,110*km)
Bmx_g,Bmy_g = magnetic_single_m(x,y,yym*km,zz*RE,jmy,jmz,110*km)

Bx = Bpx + Bhx + Bmx_g
By = Bpy + Bhy + Bmy_g

# above ionosphere, need Bmp only
Bmx,Bmy = magnetic_single_m(x,y,yym*km,zz*RE,jmy,jmz,110*km-400*km)

dot_product = Bx*Bmx + By*Bmy
B_modulus = np.sqrt(Bx**2+By**2)
Bm_modulus = np.sqrt(Bmx**2+Bmy**2)

cos_theta = dot_product/(B_modulus*Bm_modulus)
theta = np.arccos(cos_theta)*(180/np.pi)

print(theta)

plt.quiver(x,y,Bx,-By,label='$\delta B_{hor}(0km)$',color='black')
plt.quiver(x,y,Bmx,-Bmy,label='$\delta B_{MP}(400km)$',color='dodgerblue')
plt.show()