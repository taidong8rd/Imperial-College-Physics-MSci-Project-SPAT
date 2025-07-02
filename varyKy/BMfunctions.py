# -*- coding: utf-8 -*-
"""
BOX MODEL FUNCTIONS

Created on Mon Nov 21 13:50:09 2022

@author: Song Zhang
"""

import numpy as np

# fixed constants
RE = 6371*1E3  # Earth radius in meters
z0 = 25*RE
km = 1E3  # coding in units of km
kz = (2*np.pi)/(50*RE)

# variables: ky, N, sigma_P, sigma_H


def delta_jr(Y,ky,N=1): # - imaginary part of z-component of delta_J
    '''field aligned current
       evaluated along y axis
       N: number of oscillations'''
    delta_jr_list = []
    for y in Y:
        val = ky*kz*np.sin(ky*y)*np.cos(kz*z0) if abs(y) < N*np.pi/ky else 0
        delta_jr_list.append(val)
    delta_jr_list = np.array(delta_jr_list)
    
    return delta_jr_list


def delta_J_im(yy,zz,ky,N=1): # should use imaginary part
    '''magnetopause current
       evaluated at every point on an input y-z grid'''
    n,l = yy.shape
    jmy = np.zeros((n,l))
    jmz = np.zeros((n,l))
    for j in range(n):
        for k in range(l):           
            jmy[j][k] = kz**2*np.sin(kz*zz[j][k])*np.cos(ky*yy[j][k]) if abs(yy[j][k]) < N*np.pi/ky else 0
            jmz[j][k] = -ky*kz*np.sin(ky*yy[j][k])*np.cos(kz*zz[j][k]) if abs(yy[j][k]) < N*np.pi/ky else 0   
    
    return jmy,jmz


def potential_single(x0,y0,Y,ky,N=1,sigma_P=5):
    '''calculate potential at a single point (x0,y0)''' 
    delta_jr_list = delta_jr(Y,ky,N)
    integrand = delta_jr_list*np.log(np.sqrt(x0**2+(y0-Y)**2))
    
    return np.trapz(integrand,Y)/(2*np.pi*sigma_P)


def potential(xx,yy,ky,N=1,sigma_P=5):
    '''calculate potential on an input x-y grid'''
    m,n = xx.shape
    potential_list = np.zeros((m,n))
    Y = yy[:,0]
    
    delta_jr_list = delta_jr(Y,ky,N)
    
    # counter = 0
    for i in range(m):
        for j in range(n):
            integrand = delta_jr_list*np.log(np.sqrt(xx[i][j]**2+(yy[i][j]-Y)**2))
            potential_list[i][j] = np.trapz(integrand,Y)/(2*np.pi*sigma_P)
            # counter += 1
            # print('potential No.',counter)
            
    return potential_list


def electric_field(xx,yy,potential_list,ky,N=1,sigma_P=5):  
    '''calculate electric field associated with the potential'''
    m,n = potential_list.shape
    Ex = np.zeros(potential_list.shape)
    Ey = np.zeros(potential_list.shape) 
    z = potential_list # call it z: easy to type!
    
    # auto work out needed parameters from input grid matrices, all in units m   
    hx = abs(xx[0][-1]-xx[0][-2])
    hy = abs(yy[-1][0]-yy[-2][0])
    Y = yy[:,0]
    # for grid points nearest to OCB, need potential at a location closer to OCB:
    hx_prime = hx/4
    
    for i in range(m): # len(x)=m, len(y)=n
        for j in range(n): # first calculate horizontal gradient (loop over all x at each fixed y)
            if j == 0: # first point: using FDS
                Ex[i][j] = (z[i][1]-z[i][0])/hx # rewrite the corresponding element in the matrix
            elif j == n-1: # last point: using BDS
                Ex[i][j] = (z[i][n-1]-z[i][n-2])/hx
            elif j == (n/2)-1: # the last point on LHS of OCB: CDS with correction
                val = potential_single(xx[i][j]+hx_prime,yy[i][j],Y,ky,N,sigma_P)
                Ex[i][j] = (val-z[i][j-1])/(hx+hx_prime)
            elif j == n/2: # the last point on RHS of OCB: CDS with correction
                val = potential_single(xx[i][j]-hx_prime,yy[i][j],Y,ky,N,sigma_P)
                Ex[i][j] = (z[i][j+1]-val)/(hx+hx_prime)                
            else: # middle points: using CDS
                Ex[i][j] = (z[i][j+1]-z[i][j-1])/(2*hx)
    
    for j in range(n):
        for i in range(m): # then calculate vertical gredient (loop over all y at each fixed x)
            if i == 0: # FDS
                Ey[i][j] = (z[1][j]-z[0][j])/hy
            elif i == m-1: # BDS
                Ey[i][j] = (z[m-1][j]-z[m-2][j])/hy
            else: # CDS
                Ey[i][j] = (z[i+1][j]-z[i-1][j])/(2*hy)
    
    return -Ex,-Ey # E = -grad(psi)


def pedersen(Ex,Ey,sigma_P=5):
    '''Pedersen current derived from the electric field'''
    jpx = sigma_P*Ex
    jpy = sigma_P*Ey
    return jpx,jpy


def hall(Ex,Ey,sigma_H=5):
    '''Hall current derived from the electric field'''
    jhx = -sigma_H*Ey
    jhy = sigma_H*Ex
    return jhx,jhy


def magnetic_single_ph(x0,y0,xx,yy,jx,jy,h):
    '''calculate magnetic field at a single point (x0,y0)
       due to Pedersen/Hall current'''
    X = xx[0]
    Y = yy[:,0]
    
    # x0,y0,h: constants; xx,yy: integration variables; jx,y=jx,y(xx,yy)
    integrand_x = (jy*h)/((x0-xx)**2+(y0-yy)**2+h**2)**(3/2)
    integrand_y = (-jx*h)/((x0-xx)**2+(y0-yy)**2+h**2)**(3/2)    
    
    # B at (x0,y0) is calculated using all x-y grid points
    Bx = np.trapz(np.trapz(integrand_x,X),Y)
    By = np.trapz(np.trapz(integrand_y,X),Y)
    
    return Bx,By

    
def magnetic_field_ph(xx_,yy_,xx,yy,jx,jy,h):
    '''magnetic field perturbation from Pedersen/Hall current
       xx_,yy_: grid points where I want to calculate the magnetic field
       xx,yy: the entire ionospheric grid
       jx,jy: Pedersen/Hall currents'''
    m,n = xx_.shape
    Bx_list = np.zeros((m,n))
    By_list = np.zeros((m,n))
    counter = 0
    for i in range(m):
        for j in range(n): # loop over all positions of interests
            Bx,By = magnetic_single_ph(xx_[i][j],yy_[i][j],xx,yy,jx,jy,h)
            Bx_list[i][j] = Bx
            By_list[i][j] = By
            counter += 1
            print('Bph No.',counter)
            
    return Bx_list,By_list


def magnetic_single_m(x0,y0,yym,zz,jy,jz,h):
    '''calculate magnetic field at a single point (x0,y0)
       due to magnetopause current'''
    Ym = yym[0]
    Z = zz[:,0]
    z = z0 + h # constant
    
    integrand_x = (jy*(z-zz)-jz*(y0-yym))/(x0**2+(y0-yym)**2+(z-zz)**2)**(3/2)
    integrand_y = (jz*x0)/(x0**2+(y0-yym)**2+(z-zz)**2)**(3/2)
    
    Bx = np.trapz(np.trapz(integrand_x,Ym),Z)
    By = np.trapz(np.trapz(integrand_y,Ym),Z)
    
    return Bx,By


def magnetic_field_m(xx_,yy_,yym,zz,jy,jz,h):
    '''magnetic field perturbation from magnetopause current
       xx_,yy_: grid points where I want to calculate the magnetic field
       yym,zz: the entire magnetopause grid
       jx,jy: magnetopause current'''
    m,n = xx_.shape
    Bx_list = np.zeros((m,n))
    By_list = np.zeros((m,n))
    counter = 0
    for i in range(m):
        for j in range(n):
            Bx,By = magnetic_single_m(xx_[i][j],yy_[i][j],yym,zz,jy,jz,h)
            Bx_list[i][j] = Bx
            By_list[i][j] = By
            counter += 1
            print('Bm No.',counter)
            
    return Bx_list,By_list