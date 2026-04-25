# Utility functions for Monte-Carlo transport

import numpy as np
from numpy.random import rand
from scipy.constants import pi

def isotropic_direction():
    
    accept = False
    
    while not accept:
    
        u, v = 2*rand(2) - 1
        rhosq = u*u + v*v
        
        if rhosq < 1:
            accept = True
            
    nx = 1 - 2*rhosq
    ny = 2*u * np.sqrt(1 - rhosq)
    nz = 2*v * np.sqrt(1 - rhosq)
    
    return np.array([nx, ny, nz])

def isotropic_direction_in_angle(alpha: float):
    
    cosalpha = np.cos(alpha)
    nz = cosalpha + (1 - cosalpha)*rand()
    
    theta = np.acos(nz)
    beta = 2*pi*rand()
    sintheta = np.sin(theta)
    
    nx = sintheta*np.cos(beta)
    ny = sintheta*np.sin(beta)
    
    return np.array([nx, ny, nz])

def photon_direction(angle: float):
    
    nz = np.cos(angle)
    
    rho = np.sin(angle)
    phi = 2*pi*rand()
    
    nx = rho*np.cos(phi)
    ny = rho*np.sin(phi)
    
    return np.array([nx, ny, nz])

def photon_angle(energy_in):
    
    # energy_in in MeV
    a = energy_in / 0.511
    b = 1/a
    c = 1 + 2*a
    d = c/(9 + 2*a)
    
    accept = False
    
    while not accept:
        
        r1, r2, r3 = rand(3)
        e = 1 + 2*a*r1
        
        if r2 <= d:
            f = e
            g = 4*(1/f - 1/(f*f))
        else:
            f = c/e
            g = 0.5*((1 + b - b*f)**2 + 1/f)
            
        if r3 < g:
            accept = True
            
    angle = np.acos(1 + b - b*f)
    energy_out = energy_in/f
    
    return energy_out, angle

def compton_scatter(energy_in, direction):
    
    energy_out_mev, angle = photon_angle(energy_in*1e-3)
    dir_oldframe = photon_direction(angle)

    dir_newframe = transform_direction(dir_oldframe, direction)

    return energy_out_mev*1e3, dir_newframe

def transform_direction(direction, axis):
    
    ax, ay, az = axis
    s = np.sqrt(ax*ax + ay*ay)
    
    if s < 1e-3:
        
        if az > 0:
            return direction
        else:
            return -direction
    else:
        
        rot_matrix = np.array([
            [ay/s, ax*az/s, ax],
            [-ax/s, ay*az/s, ay],
            [0, -s, az]
        ])
        
        return rot_matrix @ direction
    
def intersect_plane(position, direction, planez):
    
    rz = position[2]
    nz = direction[2]
    
    if abs(nz) < 1e-3:
        dist = np.inf
    else:
        dist = (planez - rz) / nz
        
    return dist

def intersect_cylinder(position, direction, radius):

    rx, ry = position[:2]
    nx, ny = direction[:2]
    
    a = nx*nx + ny*ny
    b = 2*(rx*nx + ry*ny)
    c = rx*rx + ry*ry - radius*radius
    d = b*b - 4*a*c

    if d < 0:
        s1 = np.inf
        s2 = np.inf
    else:
        s1 = (-b + np.sqrt(d))/(2*a)
        s2 = (-b - np.sqrt(d))/(2*a)

    return [s1, s2]
        
def intersect_cylinder_in(position, direction, pztop, pzbottom, radius):

    d1, d2 = intersect_cylinder(position, direction, radius)

    dplus = intersect_plane(position, direction, pztop)
    dminus = intersect_plane(position, direction, pzbottom)

    return min(max(d1, d2), max(dplus, dminus))

def intersect_cylinder_out(position, direction, pztop, pzbottom, radius):
    
    rx, ry, rz = position
    nx, ny, nz = direction
    
    d1, d2 = intersect_cylinder(position, direction, radius)

    dmin = min(d1, d2)
    dmax = max(d1, d2)

    if dmin == np.inf or dmax < 0 or ((rz > pztop and nz > 0) or (rz < pzbottom and nz < 0)):
        dist = np.inf
    else:

        if d1*d2 > 0:
            dcyl = dmin
        else:
            dcyl = dmax

        zcyl = rz + dcyl*nz

        if zcyl < pzbottom or zcyl > pztop:
            dcyl = np.inf

        dplus = intersect_plane(position, direction, pztop)
        xplus = rx + dplus*nx
        yplus = ry + dplus*ny

        if xplus*xplus + yplus*yplus > radius*radius:
            dplus = np.inf

        dminus = intersect_plane(position, direction, pzbottom)
        xminus = rx + dminus*nx
        yminus = ry + dminus*ny

        if xminus*xminus + yminus*yminus > radius*radius:
            dminus = np.inf
            
        dist = min([dminus, dplus, dcyl]);
        
    return dist