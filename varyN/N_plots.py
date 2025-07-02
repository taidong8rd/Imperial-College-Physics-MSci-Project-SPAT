# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:12:03 2023

@author: sz3119
"""

import numpy as np
import matplotlib.pyplot as plt
import glob as glob
from BMfunctions import *
import time
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

N = 2
ky = (2*np.pi)/((2500/4)*km)
#%% create grids
X = np.linspace(-2500,2500,1600) # km
Y = np.linspace(-2500,2500,1600)
xx,yy = np.meshgrid(X,Y) # ionosphere grid

Ym = np.linspace(-2500,2500,1600)
Z = np.linspace(0,25,1600) # RE
yym,zz = np.meshgrid(Ym,Z) # magnetopause gird, might differ in Y scale
#%% plot FAC
plt.figure('FAC')
plt.plot(Y,delta_jr(Y*km,ky,N))
plt.xlabel('$y(km)$')
plt.ylabel('$\delta j_{r}$')
plt.xticks(np.arange(-2500,3000,1250))
plt.title('$\delta j_{r}$ at $z=z_{0}=25R_{E}$ as a function of y')
plt.grid()
plt.show()
#%% plot magnetopause current
jmy,jmz = delta_J_im(yym*km,zz*RE,ky,N)

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
#%% calculate potential and electric field
t1 = time.time()
# psi_list = np.loadtxt('1M\psi_N=%s_1kx1k.txt'%(N))
psi_list = np.loadtxt('2.56M\psi_N=%s_1.6kx1.6k.txt'%(N))
Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N)
t2 = time.time()
print('time used:',t2-t1)
#%% calculate currents
t1 = time.time()
jpx,jpy = pedersen(Ex,Ey)
jhx,jhy = hall(Ex,Ey)
jmy,jmz = delta_J_im(yym*km,zz*RE,ky,N)
t2 = time.time()
print('time used:',t2-t1)
#%% plot potential
plt.figure('potential',figsize=(9.6,7.2))
plt.contourf(xx,yy,psi_list,300,cmap='RdBu_r')
ticks = np.linspace(np.max(psi_list),np.min(psi_list),9)
plt.colorbar(ticks=ticks)
plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.yticks(np.arange(-2500,2501,500))
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
plt.quiver(_xx,_yy,_Ex,_Ey)
plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.yticks(np.arange(-2500,2501,500))
plt.xlim(-2500,2500)
plt.ylim(-2500,2500)
plt.title('Electric field at N-ionosphere ($z_{0}=25R_{E}$)')
plt.show()
#%% plot Pedersen/Hall and potential
plt.figure('current+potential',dpi=150)

a = 590
b = -a
c = 1
_xx = xx[a:b:c,a:b:c]
_yy = yy[a:b:c,a:b:c]
_jpx = jpx[a:b:c,a:b:c]
_jpy = jpy[a:b:c,a:b:c]
_jhx = jhx[a:b:c,a:b:c]
_jhy = jhy[a:b:c,a:b:c]

modulusP = np.sqrt(_jpx**2+_jpy**2)/np.max(np.sqrt(_jpx**2+_jpy**2))
plt.streamplot(_xx,_yy,_jpx,_jpy,cmap='BuPu',color=modulusP,
                norm=colors.LogNorm(vmin=modulusP.min(),vmax=modulusP.max()),
                arrowstyle='fancy')
plt.colorbar().ax.set_title('$|\delta \mathbf{j}_{P}|$')
plt.title('Pedersen current at northern ionosphere (N=%s)'%N)

# modulusH = np.sqrt(_jhx**2+_jhy**2)/np.max(np.sqrt(_jhx**2+_jhy**2))
# plt.streamplot(_xx,_yy,_jhx,_jhy,cmap='YlGn',color=modulusH,
#                 norm=colors.LogNorm(vmin=modulusH.min(),vmax=modulusH.max()),
#                 arrowstyle='fancy')
# plt.colorbar().ax.set_title('$|\delta \mathbf{j}_{H}|$')
# plt.title('Hall current at northern ionosphere (N=%s)'%N)

plt.contourf(xx,yy,psi_list,100,cmap='RdBu_r')
# ticks = np.linspace(np.min(psi_list),np.max(psi_list),7)
# plt.colorbar(ticks=ticks)

plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xlim(X[a],X[-a-1])
plt.ylim(Y[-a],Y[a-1])
# plt.xticks(np.arange(-2500,2501,500))
# plt.yticks(np.arange(-2500,2501,500))
# plt.xlim(-2500,2500)
# plt.ylim(2500,-2500)
plt.show()
#%% subplots above
fig = plt.figure(figsize=(13, 5),dpi=150)
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.04, 1])

a = 590
b = -a
c = 1
_xx = xx[a:b:c,a:b:c]
_yy = yy[a:b:c,a:b:c]
_jpx = jpx[a:b:c,a:b:c]
_jpy = jpy[a:b:c,a:b:c]
_jhx = jhx[a:b:c,a:b:c]
_jhy = jhy[a:b:c,a:b:c]


ax1 = plt.subplot(gs[0])
modulusP = np.sqrt(_jpx**2+_jpy**2)/np.max(np.sqrt(_jpx**2+_jpy**2))
s = ax1.streamplot(_xx,_yy,_jpx,_jpy,cmap='BuPu',color=modulusP,
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

ax1.set_title('Pedersen current at northern ionosphere (N=%s)'%N)
ax1.set_xlabel('$x(km)$')
ax1.set_ylabel('$y(km)$')
ax1.set_xlim(X[a],X[-a-1])
ax1.set_ylim(Y[-a],Y[a-1])


ax2 = plt.subplot(gs[2])
modulusH = np.sqrt(_jhx**2+_jhy**2)/np.max(np.sqrt(_jhx**2+_jhy**2))
s = plt.streamplot(_xx,_yy,_jhx,_jhy,cmap='YlGn',color=modulusH,
                norm=colors.LogNorm(vmin=modulusH.min(),vmax=modulusH.max()),
                arrowstyle='fancy')
cbar1 = plt.colorbar(s.lines,location='left',pad=0.05)
cbar1.ax.set_title('$|\delta \mathbf{j}_{H}|$')

c = ax2.contourf(xx,yy,psi_list,1000,cmap='coolwarm')

ax2.set_title('Hall current at northern ionosphere (N=%s)'%N)
ax2.set_xlabel('$x(km)$')
ax2.set_ylabel('$y(km)$')
ax2.set_xlim(X[a],X[-a-1])
ax2.set_ylim(Y[-a],Y[a-1])
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()


fig.tight_layout(w_pad=0)
#%% plot Pedersen current separately
plt.figure('Pedersen current')

a = 600
b = -a
c = 1
_xx = xx[a:b:c,a:b:c]
_yy = yy[a:b:c,a:b:c]
_jpx = jpx[a:b:c,a:b:c]
_jpy = jpy[a:b:c,a:b:c]

modulus = np.sqrt(_jpx**2+_jpy**2)/np.max(np.sqrt(_jpx**2+_jpy**2))
plt.streamplot(_xx,_yy,_jpx,_jpy,color=modulus)
#plt.streamplot(xx,yy,jpx,jpy,color=np.sqrt(jpx**2+jpy**2)/np.max(np.sqrt(jpx**2+jpy**2)))
plt.colorbar()

plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.yticks(np.arange(-2500,2501,500))
plt.xlim(-2500,2500)
plt.ylim(2500,-2500)
plt.title('Pedersen current at N-ionosphere ($z_{0}=25R_{E}$)')
plt.show()
#%% plot Hall current separately
plt.figure('Hall current')

a = 600
b = -a
c = 1
_xx = xx[a:b:c,a:b:c]
_yy = yy[a:b:c,a:b:c]
_jhx = jhx[a:b:c,a:b:c]
_jhy = jhy[a:b:c,a:b:c]

modulus = np.sqrt(_jhx**2+_jhy**2)/np.max(np.sqrt(_jhx**2+_jhy**2))
plt.streamplot(_xx,_yy,_jhx,_jhy,color=modulus)
#plt.streamplot(xx,yy,jhx,jhy,color=np.sqrt(jhx**2+jhy**2)/np.max(np.sqrt(jhx**2+jhy**2)))
plt.colorbar()

plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(-2500,2501,500))
plt.yticks(np.arange(-2500,2501,500))
plt.xlim(-2500,2500)
plt.ylim(2500,-2500)
plt.title('Hall current at N-ionosphere ($z_{0}=25R_{E}$)')
plt.show()
#%% create grid where I want the magnetic fields
X_ = np.linspace(200,2250,8)
Y_ = np.linspace(-2250,2250,7)
xx_,yy_ = np.meshgrid(X_,Y_)
plt.figure('grid for B')
plt.scatter(xx_,yy_)
#%% explore how a B contribution changes with N
h = 110*km
N_list = [1,2,3,4,5]

color_list = ['silver','-','grey','-','black']
# color_list = ['lime','-','limegreen','-','green']
#color_list = ['violet','-','fuchsia','-','purple']
# color_list = ['deepskyblue','-','dodgerblue','-','royalblue']
files = glob.glob('2.56M\*.txt')
# files = glob.glob('1M\*.txt')

for i in range(len(files)):
    if 2*i <= 4:
        t1 = time.time()
        
        psi_list = np.loadtxt(files[2*i])
        Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N_list[2*i])
        jpx,jpy = pedersen(Ex,Ey)
        jhx,jhy = hall(Ex,Ey)
        jmy,jmz = delta_J_im(yym*km,zz*RE,ky,N_list[2*i])
        
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
#        plt.quiver(xx_,yy_,Bpx_,-Bpy_,label='$\Sigma_{P}=$'+str(sigmaP_list[2*i]),color=color_list[2*i])
        # plt.quiver(xx_,yy_,Bhx_,-Bhy_,label='$\Sigma_{P}=$'+str(sigmaP_list[2*i]),color=color_list[2*i])
        # plt.quiver(xx_,yy_,Bmx_,-Bmy_,label='$\Sigma_{P}=$'+str(sigmaP_list[2*i]),color=color_list[2*i])
        plt.quiver(xx_,yy_,Bx_,-By_,label='N='+str(N_list[2*i]),color=color_list[2*i])
        
        t2 = time.time()
        print('N = %s done, time used:'%(N_list[2*i]),t2-t1)

plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(0,2501,500))
plt.yticks(np.arange(-3000,3001,500))
plt.xlim(-100,2800)
plt.ylim(2750,-2750)
plt.legend(loc=1)
plt.title('Ground $\delta \mathbf{B}_{hor}$ at different N')
plt.show()
#%% plot all contributions to B like before
h = 110*km
N = 1

psi_list = np.loadtxt('1M\psi_N=%s_1kx1k.txt'%(N))
Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N)
jpx,jpy = pedersen(Ex,Ey)
jhx,jhy = hall(Ex,Ey)
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
plt.quiver(xx_,yy_,Bpx_,-Bpy_,label='$\delta B_{P}$',color='fuchsia')
plt.quiver(xx_,yy_,Bhx_,-Bhy_,label='$\delta B_{H}$',color='limegreen')
plt.quiver(xx_,yy_,Bmx_,-Bmy_,label='$\delta B_{MP}$',color='deepskyblue')
plt.quiver(xx_,yy_,Bx_,-By_,label='$\delta B_{hor}$',color='black')

# plt.quiver(xx_,yy_,Bpx,-Bpy,label='$\delta B_{P}$',color='purple')
# plt.quiver(xx_,yy_,Bhx,-Bhy,label='$\delta B_{H}$',color='green')
# plt.quiver(xx_,yy_,Bmx,-Bmy,label='$\delta B_{MP}$',color='blue')
# plt.quiver(xx_,yy_,Bx,-By,label='$\delta B_{hor}$',color='black')

plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(0,2501,500))
plt.yticks(np.arange(-3000,3001,500))
plt.xlim(-100,3000)
plt.ylim(2750,-2750)
plt.legend(loc=1)
plt.title('Contributions to ground $\delta B_{hor}$ with N='+str(N))
plt.show()
#%% plot total magnetic fields at different altitudes
N = 1
H = [110*km,110*km-400*km,110*km-2.5*RE]
string = ['0km','400km','$2.5 R_{E}$']
color_list = ['black','dimgrey','silver']

psi_list = np.loadtxt('1M\psi_N=%s_1kx1k.txt'%(N))
Ex,Ey = electric_field(xx*km,yy*km,psi_list,ky,N)
jpx,jpy = pedersen(Ex,Ey)
jhx,jhy = hall(Ex,Ey)
jmy,jmz = delta_J_im(yym*km,zz*RE,ky,N)

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

    plt.figure('B_altitudes')
    plt.quiver(xx_,yy_,Bx_,-By_,label=string[i],color=color_list[i])
    
plt.xlabel('$x(km)$')
plt.ylabel('$y(km)$')
plt.xticks(np.arange(0,2501,500))
plt.yticks(np.arange(-3000,3001,500))
plt.ylim(2750,-2750)
plt.xlim(-100,3000)
plt.legend(loc=1)
plt.title('$\delta B_{hor}$ at different altitudes (N=%s)'%(N))
plt.show()
#%% test
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Generate some sample data
a = np.linspace(-5, 5, 100)
b = np.linspace(-5, 5, 100)
A, B = np.meshgrid(a, b)
Z1 = np.sin(A**2 + B**2)
Z2 = np.cos(A**2 + B**2)

# Create a grid of subplots with one extra row for the colorbar
fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 0.05, 1])

# Create the first subplot
ax1 = plt.subplot(gs[0])
im1 = ax1.imshow(Z1, cmap='viridis')
ax1.set_title('Subplot 1')

# Create the second subplot
ax2 = plt.subplot(gs[2])
im2 = ax2.imshow(Z2, cmap='viridis')
ax2.set_title('Subplot 2')

# Add a colorbar in the extra column
cbar_ax = plt.subplot(gs[1])
cbar = plt.colorbar(im2, cax=cbar_ax)

# Set the label for the colorbar
cbar.ax.set_ylabel("Intensity")

# Adjust the layout of the subplots
fig.tight_layout()

# Show the plot
plt.show()

