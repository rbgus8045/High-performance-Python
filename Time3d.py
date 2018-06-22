import numpy as np
from numba import jit
import time
 
@jit
def diff_o8(up,uo,um,vel,dt,h,nx,ny,nz,ne,dimx,dimy,dimz,dimxy):
    for iz in range(0,nz):
        for iy in range(0,ny):
            for ix in range(0,nx):
                i=(iz+ne)*dimxy+(iy+ne)*dimx+(ix+ne)
                up[i]=vel[i]**2*(dt/h)**2*(-615./72.*uo[i]+\
					(8./5.*(uo[i+1]+uo[i-1]+uo[i+dimx]+uo[i-dimx]+uo[i+dimxy]+uo[i-dimxy]))+\
					(-1./5.*(uo[i+2]+uo[i-2]+uo[i+2*dimx]+uo[i-2*dimx]+uo[i+2*dimxy]+uo[i-2*dimxy]))+\
					(8./315.*(uo[i+3]+uo[i-3]+uo[i+3*dimx]+uo[i-3*dimx]+uo[i+3*dimxy]+uo[i-3*dimxy]))+\
					(-1./560.*(uo[i+4]+uo[i-4]+uo[i+4*dimx]+uo[i-4*dimx]+uo[i+4*dimxy]+uo[i-4*dimxy])))+2.*uo[i]-um[i]

@jit
def inject_source(up,srcx,srcy,srcz,ne,wit,vel,dt,dimx,dimxy):
    i=(srcz+ne)*dimxy+(srcy+ne)*dimx+(srcx+ne)
    up[i] += wit*vel[i]**2*dt**2

@jit
def boundary(up,uo,um,vel,h,dt,nx,ny,nz,ne,dimx,dimy,dimz,dimxy):
    # right
    for iz in range(ne,nz+ne):
        for iy in range(ne,ny+ne):
            for ix in range(nx+ne,dimx):
                i=iz*dimxy+iy*dimx+ix
                up[i]=-vel[i]**2*dt**2/(ct1*ct2)*((uo[i]-2.*uo[i-1]+uo[i-2])/(h**2)+\
						(ct1+ct2)/(vel[i]*h*dt) * ((uo[i]-uo[i-1])-(um[i]-um[i-1])))+2.*uo[i]-um[i]
   # left
    for iz in range(ne,nz+ne):
        for iy in range(ne,ny+ne):
            for ix in range(ne,-1,-1):
                i=iz*dimxy+iy*dimx+ix
                up[i]=-vel[i]**2*dt**2/(ct1*ct2)*((uo[i]-2.*uo[i+1]+uo[i+2])/(h**2)+\
						(ct1+ct2)/(vel[i]*h*dt) * ((uo[i]-uo[i+1])-(um[i]-um[i+1])))+2.*uo[i]-um[i]    
    # front
    for iz in range(ne,nz+ne):
        for iy in range(ne,-1,-1):
            for ix in range(dimx):
                i=iz*dimxy+iy*dimx+ix
                up[i]=-vel[i]**2*dt**2/(ct1*ct2)*((uo[i]-2.*uo[i+dimx]+uo[i+2*dimx])/(h**2)+\
						(ct1+ct2)/(vel[i]*h*dt) * ((uo[i]-uo[i+dimx])-(um[i]-um[i+dimx])))+2.*uo[i]-um[i]                     
    # back
    for iz in range(ne,nz+ne):
        for iy in range(ny+ne,dimy):
            for ix in range(dimx):
                i=iz*dimxy+iy*dimx+ix
                up[i]=-vel[i]**2*dt**2/(ct1*ct2)*((uo[i]-2.*uo[i-dimx]+uo[i-2*dimx])/(h**2)+\
						(ct1+ct2)/(vel[i]*h*dt) * ((uo[i]-uo[i-dimx])-(um[i]-um[i-dimx])))+2.*uo[i]-um[i]
    # bottom
    for iz in range(nz+ne,dimz):
        for iy in range(dimy):
            for ix in range(dimx):
                i=iz*dimxy+iy*dimx+ix
                up[i]=-vel[i]**2*dt**2/(ct1*ct2)*((uo[i]-2.*uo[i-dimxy]+uo[i-2*dimxy])/(h**2)+\
						(ct1+ct2)/(vel[i]*h*dt) * ((uo[i]-uo[i-dimxy])-(um[i]-um[i-dimxy])))+2.*uo[i]-um[i] 

@jit
def time3d(vel,dt,h,nt,nx,ny,nz,ne,srcx,srcy,srcz,w,dimx,dimy,dimz,dimxy):
    
    nt=len(w)
    uo=np.zeros(dimxyz,dtype=np.float32)
    um=np.zeros(dimxyz,dtype=np.float32)
    up=np.zeros(dimxyz,dtype=np.float32)
    seismo=np.zeros((nt,nx),dtype=np.float32)

    for it in range(nt):
        diff_o8(up,uo,um,vel,dt,h,nx,ny,nz,ne,dimx,dimy,dimz,dimxy)
        inject_source(up,srcx,srcy,srcz,ne,w[it],vel,dt,dimx,dimxy)
        boundary(up,uo,um,vel,h,dt,nx,ny,nz,ne,dimx,dimy,dimz,dimxy)
        um,uo,up=uo,up,um

        for ix in range(nx):
            seismo[it,ix]=up[(srcz+ne)*dimxy+(srcy+ne)*dimx+(ne+ix)]
        if it % 100 ==0:
            print("it=%d, nt=%d"%(it,nt))

    return seismo

w=np.loadtxt("wavelet.txt",dtype=np.float32)
nt=len(w)
dt=np.float32(0.001)
h=np.float32(0.01)
order=8
ne=int(order/2)
nx=600
ny=600
nz=600
ct1=np.cos(np.pi/6.,dtype=np.float32)
ct2=np.cos(np.pi/12.,dtype=np.float32)
dimx=nx+2*ne
dimy=ny+2*ne
dimz=nz+2*ne
dimxy=dimx*dimy
dimxyz=dimxy*dimz

vel=np.ones(dimxyz,dtype=np.float32)*2

print("nt=%d,dt=%s,h=%s"%(nt,dt,h))
print("nx=%d,ny=%d,nz=%d"%(nx,ny,nz))
print("order=%d,ne=%d"%(order,ne))

srcx=int(nx/2)
srcy=int(ny/2)
srcz=1

print("Compile")
print(time.ctime())
time3d(vel,dt,h,nt,nx,ny,nz,ne,srcx,srcy,srcz,w[:5],dimx,dimy,dimz,dimxy)
print(time.ctime())
print("Test")
print(time.ctime())
seismo=time3d(vel,dt,h,nt,nx,ny,nz,ne,srcx,srcy,srcz,w,dimx,dimy,dimz,dimxy)
print(time.ctime())

seismo.tofile("600_py.bin")
