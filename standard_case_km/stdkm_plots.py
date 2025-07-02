# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:36:07 2023

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
X = np.linspace(-2500,2500,1000) # km
Y = np.linspace(-2500,2500,1000)
xx,yy = np.meshgrid(X,Y) # ionosphere grid

Ym = np.linspace(-2500,2500,1000)
Z = np.linspace(0,25,1000) # RE
yym,zz = np.meshgrid(Ym,Z) # magnetopause gird
#%% plot FAC
plt.figure('FAC',dpi=150)
plt.plot(Y,delta_jr(Y*km,ky))
plt.xlabel('$y(km)$')
plt.ylabel('$\delta j_{r}$')
plt.xticks(np.arange(-2500,3000,1250))
plt.title('FAC at the MI-interface')
plt.grid()
plt.show()
#%% plot magnetopause current
jmy,jmz = delta_J_im(yym*km,zz*RE,ky)

plt.figure('magnetopause current',dpi=150)
a = None
b = None
c = 37 #37,40,42,43,47
_yym = yym[a:b:c,a:b:c]
_zz = zz[a:b:c,a:b:c]
_jmy = jmy[a:b:c,a:b:c]
_jmz = jmz[a:b:c,a:b:c]
plt.quiver(_yym,_zz,_jmy,_jmz,label='$\delta \mathbf{j}_{mp}(y,z)$')
plt.plot(Y,25*np.ones(np.shape(Y)),color='blueviolet',label='MI-interface',linewidth=4)
# modulus = np.sqrt(jmy**2+jmz**2)/np.max(np.sqrt(jmy**2+jmz**2))
# plt.streamplot(yym,zz,jmy,jmz,color=modulus)
# plt.colorbar()
plt.xlabel('$y(km)$')
plt.ylabel('$z(R_{E})$')
plt.xticks(np.arange(-2500,3000,1250))
# plt.xticks(np.arange(-1500,2000,500))
plt.yticks(np.arange(0,30,5))
# plt.xlim(-1500,1500)
plt.ylim(0,25)
plt.title('Magnetopause current')
plt.legend(prop={'size':8},loc=2)
plt.show()
#%% calculate potential and electric field
t1 = time.time()
# psi_list = potential(xx*km,yy*km,ky)
# np.savetxt('psi_stdkm_2kx2k.txt',psi_list)
# psi_list = np.loadtxt('psi_stdkm_1.6kx1.6k.txt')
psi_list = np.loadtxt('psi_stdkm_1kx1k.txt')
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
#%% plot potential
plt.figure('potential',dpi=150)
plt.contourf(xx,yy,psi_list,300,cmap='RdBu_r')
ticks = np.linspace(np.max(psi_list),np.min(psi_list),9)
plt.colorbar(ticks=ticks)
plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.yticks(np.arange(-2500,2501,500))
plt.xlim(-2500,2500)
plt.ylim(2500,-2500)
plt.title('Electric potential at N-ionosphere ($z_{0}=25R_{E}$)')
plt.show()
#%% plot electric field
plt.figure('electric field',dpi=150)
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
plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.yticks(np.arange(-2500,2501,500))
plt.xlim(-2500,2500)
plt.ylim(2500,-2500)
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

plt.contourf(xx,yy,psi_list,100,cmap='RdBu_r')
# ticks = np.linspace(np.min(psi_list),np.max(psi_list),7)
# plt.colorbar(ticks=ticks)

plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xlim(-2500,2500)
plt.ylim(2500,-2500)
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

c = ax1.contourf(xx,yy,psi_list,1000,cmap='coolwarm')
cbar2_ax = plt.subplot(gs[1])
cbar2 = plt.colorbar(c, cax=cbar2_ax)
cbar2.set_ticks([np.min(psi_list),0,np.max(psi_list)])
cbar2.set_ticklabels(['-1','0','1'])
cbar2.ax.set_title('$\delta \psi_{isp}$')
cbar2.ax.invert_yaxis() 

ax1.set_title('Pedersen current at northern ionosphere')
ax1.set_xlabel('$x(km)$')
ax1.set_ylabel('$y(km)$')
ax1.set_xlim(-2500,2500)
ax1.set_ylim(2500,-2500)


ax2 = plt.subplot(gs[2])
modulusH = np.sqrt(jhx**2+jhy**2)/np.max(np.sqrt(jhx**2+jhy**2))
s = plt.streamplot(xx,yy,jhx,jhy,cmap='YlGn',color=modulusH,
                norm=colors.LogNorm(vmin=modulusH.min(),vmax=modulusH.max()),
                arrowstyle='fancy')
cbar1 = plt.colorbar(s.lines,location='left',pad=0.05)
cbar1.ax.set_title('$|\delta \mathbf{j}_{H}|$')

c = ax2.contourf(xx,yy,psi_list,1000,cmap='coolwarm')

ax2.set_title('Hall current at northern ionosphere')
ax2.set_xlabel('$x(km)$')
ax2.set_ylabel('$y(km)$')
ax2.set_xlim(-2500,2500)
ax2.set_ylim(2500,-2500)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()


fig.tight_layout(w_pad=0)
#%% plot Pedersen current separately
plt.figure('Pedersen current',dpi=150)

modulus = np.sqrt(jpx**2+jpy**2)/np.max(np.sqrt(jpx**2+jpy**2))
plt.streamplot(xx,yy,jpx,jpy,color=modulus)
plt.colorbar()

plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.title('Pedersen current at N-ionosphere ($z_{0}=25R_{E}$)')
plt.show()
#%% plot Hall current separately
plt.figure('Hall current',dpi=150)

modulus = np.sqrt(jhx**2+jhy**2)/np.max(np.sqrt(jhx**2+jhy**2))
plt.streamplot(xx,yy,jhx,jhy,color=modulus)
plt.colorbar()

plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.title('Hall current at N-ionosphere ($z_{0}=25R_{E}$)')
plt.show()
#%% create grid where I want the magnetic fields
X_ = np.linspace(200,2250,8)
Y_ = np.linspace(-2250,2250,7)

# X_ = np.linspace(-2000,2000,10)
# Y_ = np.linspace(-2000,2000,9)

xx_,yy_ = np.meshgrid(X_,Y_)
plt.figure('grid for B')
plt.scatter(xx_,yy_)
#%% plot all contributions to B
h = 110*km
# h = 110*km-400*km

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
plt.quiver(xx_,yy_,Bpx_,-Bpy_,label='$\delta \mathbf{B}_{P}$',color='fuchsia')
plt.quiver(xx_,yy_,Bhx_,-Bhy_,label='$\delta \mathbf{B}_{H}$',color='limegreen')
plt.quiver(xx_,yy_,Bmx_,-Bmy_,label='$\delta \mathbf{B}_{mp}$',color='deepskyblue')
plt.quiver(xx_,yy_,Bx_,-By_,label='$\delta \mathbf{B}_{hor}$',color='black')

plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(0,2501,500))
plt.yticks(np.arange(-3000,3001,500))
# plt.xlim(-100,2900)
plt.xlim(-100,3000)
plt.ylim(2750,-2750)
# plt.gca().invert_yaxis()
plt.legend(loc=1)
plt.title('Contributions to ground $\delta \mathbf{B}_{hor}$')
#plt.title('Contributions to $\delta \mathbf{B}_{hor}$ at 400km altitudes')
plt.show()
#%% plot magnetic fields at different altitudes
# H = [110*km,110*km-400*km,110*km-2.5*RE]
# string = ['0km','400km','$2.5 R_{E}$']
# color_list_hor = ['black','dimgrey','silver']
# color_list_H = ['lime','limegreen','green']
# color_list_P = ['violet','fuchsia','purple']
# color_list_mp = ['deepskyblue','dodgerblue','royalblue']

H = [110*km,110*km-400*km]
string = ['(0km)','(400km)']
color_list_hor = ['black','silver']
# color_list_H = ['lime','limegreen','green']
# color_list_P = ['violet','fuchsia','purple']
color_list_mp = ['royalblue','deepskyblue']

for i in range(len(H)):
    Bpx,Bpy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jpx,jpy,H[i])
    Bhx,Bhy = magnetic_field_ph(xx_*km,yy_*km,xx*km,yy*km,jhx,jhy,H[i])
    Bmx,Bmy = magnetic_field_m(xx_*km,yy_*km,yym*km,zz*RE,jmy,jmz,H[i])
    
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
    plt.quiver(xx_,yy_,Bmx_,-Bmy_,label='$\delta \mathbf{B}_{mp}$'+string[i],color=color_list_mp[i])
    plt.quiver(xx_,yy_,Bx_,-By_,label='$\delta \mathbf{B}_{hor}$'+string[i],color=color_list_hor[i])
    
plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(0,2501,500))
plt.yticks(np.arange(-3000,3001,500))
plt.xlim(-100,3200)
plt.ylim(2750,-2750)
# plt.gca().invert_yaxis()
plt.legend(loc=1,prop={'size':8})
# plt.title('$\delta \mathbf{B}_{hor}$ at different altitudes')
plt.title('Magnetic fields at different altitudes')
plt.show()