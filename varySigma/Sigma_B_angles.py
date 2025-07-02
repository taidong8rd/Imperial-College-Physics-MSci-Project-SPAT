# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 01:42:29 2023

@author: zhang
"""

import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import time
from scipy.optimize import curve_fit
from BMfunctions import *
from matplotlib.colors import CenteredNorm
import matplotlib.gridspec as gridspec

N = 1
ky = (2*np.pi)/(2500*km)

#%% create grids
X = np.linspace(-2500,2500,1000)
Y = np.linspace(-2500,2500,1000)
xx,yy = np.meshgrid(X,Y) # ionosphere grid

Ym = np.linspace(-2500,2500,1000)
Z = np.linspace(0,25,1000) # RE
yym,zz = np.meshgrid(Ym,Z) # magnetopause gird

X_ = np.linspace(-2250,2250,18)
Y_ = np.linspace(-2250,2250,17)
xx_,yy_ = np.meshgrid(X_,Y_) # grid pts where I want magnetic fields
plt.figure('grid for B')
plt.scatter(xx_,yy_)
#%% calculate Bhor(ground) and Bmp(400km) only
sigmaP = 5
sigmaH = 5

t1 = time.time()

# psi_list = potential(xx*km,yy*km,ky,N,sigmaP)
# np.savetxt('1M\psi_%sS_1kx1k.txt'%(sigmaP),psi_list)
# psi_list = np.loadtxt('psi_1S_1.6kx1.6k.txt')
psi_list = np.loadtxt('1M\psi_%sS_1kx1k.txt'%(sigmaP))
Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N,sigmaP)
jpx,jpy = pedersen(Ex,Ey,sigmaP)
jhx,jhy = hall(Ex,Ey,sigmaH)
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

t2 = time.time()
print('time used:',t2-t1)
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
plt.title('Magnetic fields at different altitudes with $\Sigma_{P}=%sS,\Sigma_{H}=%sS$'%(sigmaP,sigmaH))
plt.show()
#%% quiver plot and potential
fig = plt.figure(dpi=150)
gs = gridspec.GridSpec(1, 2, width_ratios=[1,0.04])

ax1 = plt.subplot(gs[0])

c = ax1.contourf(xx,yy,psi_list,1000,cmap='RdBu_r')
cbar_ax = plt.subplot(gs[1])
cbar = plt.colorbar(c, cax=cbar_ax)
cbar.set_ticks([np.min(psi_list),0,np.max(psi_list)])
cbar.set_ticklabels(['-1','0','1'])
cbar.ax.invert_yaxis()
cbar.ax.set_title('$\delta \psi_{isp}$')

ax1.quiver(xx_,yy_,Bx_,-By_,label='$\delta \mathbf{B}_{hor}(0km)$',color='black')
ax1.quiver(xx_,yy_,Bmx_,-Bmy_,label='$\delta \mathbf{B}_{MP}(400km)$',color='blue')

ax1.set_title('Magnetic fields at different altitudes ($\Sigma_{P}=%sS,\Sigma_{H}=%sS$)'%(sigmaP,sigmaH))
ax1.set_xlabel('$x(km)$')
ax1.set_ylabel('$y(km)$')
ax1.legend(loc=2,prop={'size':8})
ax1.invert_yaxis()

gs.tight_layout(fig)
#%% work out angle between them
dot_product = Bx*Bmx + By*Bmy
B_modulus = np.sqrt(Bx**2+By**2)
Bm_modulus = np.sqrt(Bmx**2+Bmy**2)

cos_theta = dot_product/(B_modulus*Bm_modulus)
theta = np.arccos(cos_theta)*(180/np.pi)
print(theta)
#%% color plot for the angles
name = 'angles_highres_OCB_P%sH%s'%(sigmaP,sigmaH)
plt.figure(name,dpi=150)
plt.contourf(xx_,yy_,theta,norm=CenteredNorm(90),levels=1000,cmap='seismic')
ticks = np.array([np.min(theta),(np.min(theta)+90)/2,90,(np.max(theta)+90)/2,np.max(theta)])
plt.colorbar(ticks=ticks,format='%.1f').ax.set_title('Degree($\degree$)',size=10,pad=7)
plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.gca().invert_yaxis()
plt.title('Rotation angles ($\Sigma_{P}=%sS,\Sigma_{H}=%sS$)'%(sigmaP,sigmaH))
plt.savefig(name)
plt.show()
#%% do more at a time (change sigmaP)
sigmaH = 5
sigmaP_list = [1,10]

for sigmaP in sigmaP_list:
    psi_list = np.loadtxt('1M\psi_%sS_1kx1k.txt'%(sigmaP))
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N,sigmaP)
    jpx,jpy = pedersen(Ex,Ey,sigmaP)
    jhx,jhy = hall(Ex,Ey,sigmaH)
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
 
    name = 'angles_highres_OCB_P%sH%s'%(sigmaP,sigmaH)
    plt.figure(name,dpi=150)
    plt.contourf(xx_,yy_,theta,norm=CenteredNorm(90),levels=1000,cmap='seismic')
    ticks = np.array([np.min(theta),(np.min(theta)+90)/2,90,(np.max(theta)+90)/2,np.max(theta)])
    plt.colorbar(ticks=ticks,format='%.1f').ax.set_title('Degree($\degree$)',size=10,pad=7)
    plt.xlabel('$x(km)$')
    plt.ylabel('$y(km)$')
    plt.gca().invert_yaxis()
    plt.title('Rotation angles ($\Sigma_{P}=%sS,\Sigma_{H}=%sS$)'%(sigmaP,sigmaH))
    plt.savefig(name)
    # plt.show()
#%% do more at a time (change sigmaH)
sigmaP = 5
sigmaH_list = [1,5,10]

for sigmaH in sigmaH_list:
    psi_list = np.loadtxt('1M\psi_%sS_1kx1k.txt'%(sigmaP))
    Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N,sigmaP)
    jpx,jpy = pedersen(Ex,Ey,sigmaP)
    jhx,jhy = hall(Ex,Ey,sigmaH)
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
 
    name = 'angles_highres_OCB_P%sH%s'%(sigmaP,sigmaH)
    plt.figure(name,dpi=150)
    plt.contourf(xx_,yy_,theta,norm=CenteredNorm(90),levels=1000,cmap='seismic')
    ticks = np.array([np.min(theta),(np.min(theta)+90)/2,90,(np.max(theta)+90)/2,np.max(theta)])
    plt.colorbar(ticks=ticks,format='%.1f').ax.set_title('Degree($\degree$)',size=10,pad=7)
    plt.xlabel('$x(km)$')
    plt.ylabel('$y(km)$')
    plt.gca().invert_yaxis()
    plt.title('Rotation angles ($\Sigma_{P}=%sS,\Sigma_{H}=%sS$)'%(sigmaP,sigmaH))
    plt.savefig(name)
    # plt.show()
