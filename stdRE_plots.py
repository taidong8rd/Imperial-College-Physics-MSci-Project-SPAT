# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:37:16 2023

@author: zhang
"""

import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import time
from scipy.optimize import curve_fit
from BMfunctions import *
from matplotlib.colors import CenteredNorm
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

ky = (2*np.pi)/(10*RE)
#%% create grids
X = np.linspace(-10,10,1000) # unit RE
Y = np.linspace(-10,10,1000)
xx,yy = np.meshgrid(X,Y) # ionosphere grid

Ym = np.linspace(-10,10,1000)
Z = np.linspace(0,25,1000) # RE
yym,zz = np.meshgrid(Ym,Z) # magnetopause gird
#%% plot FAC
plt.figure('FAC')
plt.plot(Y,delta_jr(Y*RE,ky))
plt.xlabel('$y(R_{E})$')
plt.ylabel('$\delta j_{r}$')
plt.xticks(np.arange(-10,15,5))
plt.title('$\delta j_{r}$ at $z=z_{0}=25R_{E}$ as a function of y')
plt.grid()
plt.show()
#%% plot magnetopause current
jmy,jmz = delta_J_im(yym*RE,zz*RE,ky)

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
plt.xlabel('$y(R_{E})$')
plt.ylabel('$z(R_{E})$')
plt.xticks(np.arange(-10,15,5))
plt.yticks(np.arange(0,30,5))
plt.title('Magnetopause current ($\delta \mathbf{j}_{mp}$)')
plt.show()
#%% calculate potential and electric field
t1 = time.time()
# psi_list = potential(xx*RE,yy*RE,ky)
# np.savetxt('psi_stdRE_2kx2k.txt',psi_list)
psi_list = np.loadtxt('psi_stdRE_1kx1k.txt')
Ex,Ey = electric_field(xx*RE,yy*RE,psi_list,ky)
t2 = time.time()
print('time used:',t2-t1)
#%% calculate currents
t1 = time.time()
jpx,jpy = pedersen(Ex,Ey)
jhx,jhy = hall(Ex,Ey)
jmy,jmz = delta_J_im(yym*RE,zz*RE,ky)
t2 = time.time()
print('time used:',t2-t1)
#%% plot potential
plt.figure('potential',dpi=150)
plt.contourf(xx,yy,psi_list,300,cmap='RdBu_r')
ticks = np.linspace(np.max(psi_list),np.min(psi_list),9)
plt.colorbar(ticks=ticks)
plt.xlabel('$x(R_{E})$')
plt.ylabel('$y(R_{E})$')
plt.xticks(np.arange(-10,12,2))
plt.yticks(np.arange(-10,12,2))
plt.xlim(-10,10)
plt.ylim(10,-10)
plt.title('Electric potential at N-ionosphere ($z_{0}=25R_{E}$)')
plt.show()
#%% plot electric field
plt.figure('electric field')
# plt.quiver(xx,yy,Ex,Ey)
a = None
b = None
c = 40
_xx = xx[a:b:c,a:b:c]
_yy = yy[a:b:c,a:b:c]
_Ex = Ex[a:b:c,a:b:c]
_Ey = Ey[a:b:c,a:b:c]
# modulus = np.sqrt(_Ex**2+_Ey**2)
# plt.streamplot(_xx,_yy,_Ex,_Ey,color=modulus)
# plt.colorbar()
plt.quiver(_xx,_yy,_Ex,-_Ey)
plt.xlabel('$x(R_{E})$')
plt.ylabel('$y(R_{E})$')
plt.xticks(np.arange(-10,12,2))
plt.yticks(np.arange(-10,12,2))
plt.xlim(-10,10)
plt.ylim(10,-10)
plt.title('Electric field at N-ionosphere ($z_{0}=25R_{E}$)')
plt.show()
#%% plot Pedersen/Hall and potential
plt.figure('current+potential',dpi=150)

modulusP = np.sqrt(jpx**2+jpy**2)/np.max(np.sqrt(jpx**2+jpy**2))
plt.streamplot(xx,yy,jpx,jpy,cmap='BuPu',color=modulusP,
                norm=colors.LogNorm(vmin=modulusP.min(),vmax=modulusP.max()),
                arrowstyle='fancy')
plt.colorbar().ax.set_title('$|\delta \mathbf{j}_{P}|$')
plt.title('Pedersen current at northern ionosphere')

# modulusH = np.sqrt(jhx**2+jhy**2)/np.max(np.sqrt(jhx**2+jhy**2))
# plt.streamplot(xx,yy,jhx,jhy,cmap='YlGn',color=modulusH,
#                 norm=colors.LogNorm(vmin=modulusH.min(),vmax=modulusH.max()),
#                 arrowstyle='fancy')
# plt.colorbar().ax.set_title('$|\delta \mathbf{j}_{H}|$')
# plt.title('Hall current at northern ionosphere')

plt.contourf(xx,yy,psi_list,1000,cmap='RdBu_r')
# ticks = np.linspace(np.min(psi_list),np.max(psi_list),7)
# plt.colorbar(ticks=ticks)

plt.xlabel('$x(R_{E})$')
plt.ylabel('$y(R_{E})$')
plt.xticks(np.arange(-10,12,2))
plt.yticks(np.arange(-10,12,2))
plt.xlim(-10,10)
plt.ylim(10,-10)
plt.show()
#%% subplots above
fig = plt.figure(figsize=(13, 5),dpi=150)
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.04, 1])

ax1 = plt.subplot(gs[0])
modulusP = np.sqrt(jpx**2+jpy**2)/np.max(np.sqrt(jpx**2+jpy**2))
s = ax1.streamplot(xx,yy,jpx,jpy,cmap='BuPu',color=modulusP,
                norm=colors.LogNorm(vmin=modulusP.min(),vmax=modulusP.max()),
                arrowstyle='fancy')
cbar1 = plt.colorbar(s.lines,location='right')
cbar1.ax.set_title('$|\delta \mathbf{j}_{P}|$')

c = ax1.contourf(xx,yy,psi_list,1000,cmap='RdBu_r')
cbar2_ax = plt.subplot(gs[1])
cbar2 = plt.colorbar(c, cax=cbar2_ax)
cbar2.set_ticks([np.min(psi_list),0,np.max(psi_list)])
cbar2.set_ticklabels(['-1','0','1'])
cbar2.ax.set_title('$\delta \psi_{isp}$')
cbar2.ax.invert_yaxis()

ax1.set_title('Pedersen current at northern ionosphere')
ax1.set_xlabel('$x(R_{E})$')
ax1.set_ylabel('$y(R_{E})$')
ax1.set_xlim(-10,10)
ax1.set_ylim(10,-10)


ax2 = plt.subplot(gs[2])
modulusH = np.sqrt(jhx**2+jhy**2)/np.max(np.sqrt(jhx**2+jhy**2))
s = plt.streamplot(xx,yy,jhx,jhy,cmap='YlGn',color=modulusH,
                norm=colors.LogNorm(vmin=modulusH.min(),vmax=modulusH.max()),
                arrowstyle='fancy')
cbar1 = plt.colorbar(s.lines,location='left',pad=0.05)
cbar1.ax.set_title('$|\delta \mathbf{j}_{H}|$')

c = ax2.contourf(xx,yy,psi_list,1000,cmap='RdBu_r')

ax2.set_title('Hall current at northern ionosphere')
ax2.set_xlabel('$x(R_{E})$')
ax2.set_ylabel('$y(R_{E})$')
ax2.set_xlim(-10,10)
ax2.set_ylim(10,-10)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()


fig.tight_layout(w_pad=0)
#%% create grid where I want the magnetic fields
X_ = np.linspace(0.05,8.75,8) # arbitrary positions
Y_ = np.linspace(-3.75,3.75,7)
xx_,yy_ = np.meshgrid(X_,Y_)
plt.figure('grid for B')
plt.scatter(xx_,yy_)
#%% plot all contributions to B like before
h = 110*km

Bpx,Bpy = magnetic_field_ph(xx_*RE,yy_*RE,xx*RE,yy*RE,jpx,jpy,h)
Bhx,Bhy = magnetic_field_ph(xx_*RE,yy_*RE,xx*RE,yy*RE,jhx,jhy,h)
Bmx,Bmy = magnetic_field_m(xx_*RE,yy_*RE,yym*RE,zz*RE,jmy,jmz,h)

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
plt.quiver(xx_,yy_,Bpx_,-Bpy_,label='$\delta \mathbf{B}_{P}$',color='fuchsia')
plt.quiver(xx_,yy_,Bhx_,-Bhy_,label='$\delta \mathbf{B}_{H}$',color='limegreen')
plt.quiver(xx_,yy_,Bmx_,-Bmy_,label='$\delta \mathbf{B}_{mp}$',color='deepskyblue')
plt.quiver(xx_,yy_,Bx_,-By_,label='$\delta \mathbf{B}_{hor}$',color='black')

# plt.quiver(xx_,yy_,Bpx,-Bpy,label='$\delta B_{P}$',color='purple')
# plt.quiver(xx_,yy_,Bhx,-Bhy,label='$\delta B_{H}$',color='green')
# plt.quiver(xx_,yy_,Bmx,-Bmy,label='$\delta B_{MP}$',color='blue')
# plt.quiver(xx_,yy_,Bx,-By,label='$\delta B_{hor}$',color='black')

plt.xlabel('$x(R_{E})$')
plt.ylabel('$y(R_{E})$')
plt.xticks(np.arange(-1,11,1))
plt.yticks(np.arange(-5,6,1))
plt.xlim(-1,11.5)
plt.ylim(5,-5)
plt.legend(loc=1)
plt.title('Contributions to ground $\delta \mathbf{B}_{hor}$')
plt.show()
#%% plot total magnetic fields at different altitudes
H = [110*km,110*km-400*km,110*km-2.5*RE]
string = ['0km','400km','$2.5 R_{E}$']
color_list = ['black','dimgrey','silver']

for i in range(len(H)):
    Bpx,Bpy = magnetic_field_ph(xx_*RE,yy_*RE,xx*RE,yy*RE,jpx,jpy,H[i])
    Bhx,Bhy = magnetic_field_ph(xx_*RE,yy_*RE,xx*RE,yy*RE,jhx,jhy,H[i])
    Bmx,Bmy = magnetic_field_m(xx_*RE,yy_*RE,yym*RE,zz*RE,jmy,jmz,H[i])
    
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

    plt.figure('B_altitudes',dpi=150)
    plt.quiver(xx_,yy_,Bx_,-By_,label=string[i],color=color_list[i])
    
plt.xlabel('$x(R_{E})$')
plt.ylabel('$y(R_{E})$')
plt.xticks(np.arange(-1,11,1))
plt.yticks(np.arange(-5,6,1))
plt.xlim(-1,12)
plt.ylim(5,-5)
plt.legend(loc=1)
plt.title('$\delta \mathbf{B}_{hor}$ at different altitudes')
plt.show()
