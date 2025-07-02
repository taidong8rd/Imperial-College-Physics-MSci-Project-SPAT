# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 21:51:15 2023

@author: zhang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib.colors import CenteredNorm
import time
from BMfunctions import *

ky = (2*np.pi)/(2500*km)
#%% create grids
X = np.linspace(-2500,2500,1000) # km
Y = np.linspace(-2500,2500,1000)
xx,yy = np.meshgrid(X,Y) # ionosphere grid

Ym = np.linspace(-2500,2500,1000)
Z = np.linspace(0,25,1000) # RE
yym,zz = np.meshgrid(Ym,Z) # magnetopause gird
#%% create grid where I want the magnetic fields
# X_ = np.linspace(200,2250,8)
# Y_ = np.linspace(-2250,2250,7)

# X_ = np.linspace(-1000,1000,101)
# Y_ = np.linspace(-2000,2000,101)

X_ = np.linspace(-2000,2000,9)
Y_ = np.linspace(-2000,2000,9)

xx_,yy_ = np.meshgrid(X_,Y_)
plt.figure('grid for B')
plt.scatter(xx_,yy_)
#%% calculate potential, electric field and currents
t1 = time.time()
# psi_list = potential(xx*km,yy*km,ky)
# np.savetxt('psi_stdkm_500x500.txt',psi_list)
psi_list = np.loadtxt('psi_stdkm_1kx1k.txt')
Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky)
jpx,jpy = pedersen(Ex,Ey)
jhx,jhy = hall(Ex,Ey)
jmy,jmz = delta_J_im(yym*km,zz*RE,ky)
t2 = time.time()
print('time used:',t2-t1)
#%% Bh(0), Bmp(0), Bhor(0) and Bmp(400)
h = 110*km

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

Bmx400,Bmy400 = magnetic_field_m(xx_*km,yy_*km,yym*km,zz*RE,jmy,jmz,110*km-400*km)
Bmx400_ = Bmx400/np.sqrt(Bx**2+By**2)
Bmy400_ = Bmy400/np.sqrt(Bx**2+By**2)
#%% plot
fig = plt.figure(dpi=150)
gs = gridspec.GridSpec(1, 2, width_ratios=[1,0.04])

ax1 = plt.subplot(gs[0])

c = ax1.contourf(xx,yy,psi_list,1000,cmap='coolwarm')
cbar_ax = plt.subplot(gs[1])
cbar = plt.colorbar(c, cax=cbar_ax)
cbar.set_ticks([np.min(psi_list),0,np.max(psi_list)])
cbar.set_ticklabels(['-1','0','1'])
cbar.ax.invert_yaxis()
cbar.ax.set_title('$\delta \psi_{isp}$')

ax1.quiver(xx_,yy_,Bhx_,-Bhy_,label='$\delta \mathbf{B}_{H}(0km)$',color='lime')
ax1.quiver(xx_,yy_,Bmx_,-Bmy_,label='$\delta \mathbf{B}_{mp}(0km)$',color='cyan')
ax1.quiver(xx_,yy_,Bx_,-By_,label='$\delta \mathbf{B}_{hor}(0km)$',color='black')
ax1.quiver(xx_,yy_,Bmx400_,-Bmy400_,label='$\delta \mathbf{B}_{mp}(400km)$',color='blue')

ax1.set_title('Magnetic fields at different altitudes')
ax1.set_xlabel('$x(km)$')
ax1.set_ylabel('$y(km)$')
ax1.legend(loc=2,prop={'size':6.2})
# ax1.invert_yaxis()
ax1.set_xlim(-2250,2250)
ax1.set_ylim(2250,-2250)

gs.tight_layout(fig)
#%% compute angles
def angle(B1x,B1y,B2x,B2y):
    dot_product = B1x*B2x + B1y*B2y
    modulus1 = np.sqrt(B1x**2+B1y**2)
    modulus2 = np.sqrt(B2x**2+B2y**2)
    cos_theta = dot_product/(modulus1*modulus2)
    theta = np.arccos(cos_theta)*(180/np.pi)
    return theta

theta = angle(Bx,By,Bmx400,Bmy400)
print(theta)
#%% color plot angles
plt.figure('angles',dpi=150)
# plt.contourf(xx_,yy_,theta,levels=1000,cmap='YlOrRd')
# ticks = np.linspace(np.max(theta),np.min(theta),7)
# plt.colorbar(ticks=ticks,format='%.1f').ax.set_title('Degree($\degree$)',size=10,pad=7)
plt.contourf(xx_,yy_,theta,norm=CenteredNorm(90),levels=1000,cmap='seismic')
ticks = np.array([np.min(theta),(np.min(theta)+90)/2,90,(np.max(theta)+90)/2,np.max(theta)])
plt.colorbar(ticks=ticks,format='%.1f').ax.set_title('Degree($\degree$)',size=10,pad=7)
plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.gca().invert_yaxis()
plt.title('Angle between $\delta \mathbf{B}_{mp}(400km)$ and $\delta \mathbf{B}_{hor}(0km)$')
# plt.savefig('angles')
plt.show()