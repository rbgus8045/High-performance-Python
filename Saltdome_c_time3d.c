#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void diff_o8(float *up,float *uo,float *um,float *vel,float dt,float h,int nx,int ny,int nz,int ne,int dimx,int dimy,int dimz,int dimxy)
{
	int i,ix,iy,iz;
	for (iz=0;iz<nz;iz++)
	for (iy=0;iy<ny;iy++)
		for (ix=0;ix<nx;ix++)
		{
			i=(iz+ne)*dimxy+(iy+ne)*dimx+(ix+ne);
			up[i]=vel[i]*vel[i]*(dt/h)*(dt/h)*(-615./72.*uo[i]+ \
				((-1./560.)*(uo[i-4]+uo[i-4*dimx]+uo[i-4*dimxy]+uo[i+4]+uo[i+4*dimx]+uo[i+4*dimxy])) + \
				((8./315.)*(uo[i-3]+uo[i-3*dimx]+uo[i-3*dimxy]+uo[i+3]+uo[i+3*dimx]+uo[i+3*dimxy])) + \
				((-1./5.)*(uo[i-2]+uo[i-2*dimx]+uo[i-2*dimxy]+uo[i+2]+uo[i+2*dimx]+uo[i+2*dimxy])) + \
				((8./5.)*(uo[i-1]+uo[i-dimx]+uo[i-dimxy]+uo[i+1]+uo[i+dimx]+uo[i+dimxy])))+2.*uo[i]-um[i];
		}
}

void inject_source(float *up,int srcx,int srcy,int srcz,int ne,float wit,float *vel,float dt,int dimx,int dimxy)
{
	int i=(srcz+ne)*dimxy+(srcy+ne)*dimx+(srcx+ne);
	up[i] += wit*vel[i]*vel[i]*dt*dt;
}

void boundary(float *up,float *uo,float *um,float *vel,float h,float dt,int nx,int ny,int nz,int ne,int dimx,int dimy,int dimz,int dimxy)
{
	float ct1=cos(M_PI/6.),ct2=cos(M_PI/12.);
	float uxx, uxt;
	int i, ix, iy, iz;
	// right
	for(iz=ne;iz<nz+ne;iz++)
	for(iy=ne;iy<ny+ne;iy++)
		for(ix=nx+ne;ix<dimx;ix++)
		{
			i=iz*dimxy+iy*dimx+ix;
			up[i]=-vel[i]*vel[i]*dt*dt/(ct1*ct2)*((uo[i]-2.*uo[i-1]+uo[i-2])/(h*h)+\
				(ct1+ct2)/(vel[i]*h*dt)*((uo[i]-uo[i-1])-(um[i]-um[i-1])))+2.*uo[i]-um[i];
		}
	// left
	for(iz=ne;iz<nz+ne;iz++)
	for(iy=ne;iy<ny+ne;iy++)
		for(ix=ne-1;ix>=0;ix--)
		{
			i=iz*dimxy+iy*dimx+ix;
			up[i]=-vel[i]*vel[i]*dt*dt/(ct1*ct2)*((uo[i]-2.*uo[i+1]+uo[i+2])/(h*h)+\
				(ct1+ct2)/(vel[i]*h*dt)*((uo[i]-uo[i+1])-(um[i]-um[i+1])))+2.*uo[i]-um[i];
		}
	// front
	for(iz=ne;iz<nz+ne;iz++)
	for(iy=ne-1;iy>=0;iy--)
		for(ix=0;ix<dimx;ix++)
		{
			i=iz*dimxy+iy*dimx+ix;
			up[i]=-vel[i]*vel[i]*dt*dt/(ct1*ct2)*((uo[i]-2.*uo[i+dimx]+uo[i+2*dimx])/(h*h)+(ct1+ct2)/(vel[i]*h*dt)*((uo[i]-uo[i+dimx])-(um[i]-um[i+dimx])))+2.*uo[i]-um[i];
		}
	// back
	for(iz=ne;iz<nz+ne;iz++)
	for(iy=ny+ne;iy<dimy;iy++)
		for(ix=0;ix<dimx;ix++)
		{
			i=iz*dimxy+iy*dimx+ix;
			up[i]=-vel[i]*vel[i]*dt*dt/(ct1*ct2)*((uo[i]-2.*uo[i-dimx]+uo[i-2*dimx])/(h*h)+\
				(ct1+ct2)/(vel[i]*h*dt)*((uo[i]-uo[i-dimx])-(um[i]-um[i-dimx])))+2.*uo[i]-um[i];
		}
	// bottom
	for(iz=nz+ne;iz<dimz;iz++)
	for(iy=0;iy<dimy;iy++)
		for(ix=0;ix<dimx;ix++)
		{	
			i=iz*dimxy+iy*dimx+ix;
			up[i]=-vel[i]*vel[i]*dt*dt/(ct1*ct2)*((uo[i]-2.*uo[i-dimxy]+uo[i-2*dimxy])/(h*h)+\
				(ct1+ct2)/(vel[i]*h*dt)*((uo[i]-uo[i-dimxy])-(um[i]-um[i-dimxy])))+2.*uo[i]-um[i];
		}
}

void time3d(float *seismo,float *up,float *uo,float *um,float *tmp,float *vel,int nx,int ny, int nz,int ne,float dt,float h,float *w,int srcx,int srcy,int srcz,int nt,int dimx,int dimy,int dimz,int dimxy,FILE *fp)
{
	int it,ix;
	for(it=0;it<nt;it++)
	{
		if(it%100==0)
			printf("it=%4d\n",it);
		diff_o8(up,uo,um,vel,dt,h,nx,ny,nz,ne,dimx,dimy,dimz,dimxy);
		inject_source(up,srcx,srcy,srcz,ne,w[it],vel,dt,dimx,dimxy);
		boundary(up,uo,um,vel,h,dt,nx,ny,nz,ne,dimx,dimy,dimz,dimxy);	
		for(ix=0;ix<nx;ix++)
			seismo[it*nx+ix]=up[(srcz+ne)*dimxy+(srcy+ne)*dimx+ne+ix];
		// time march
		tmp=um;
		um=uo;
		uo=up;
		up=tmp;
	}
}

int main()
{
	float *w,*up,*uo,*um,*tmp,*vel,*seismo;
	int i,it;
	int nt=3000;
	float dt=0.002;
	float h=0.02;
	int order=8;
	int ne=order/2;
    int nx=676;
	int ny=676;
	int nz=201;
	float ct1=cos(M_PI/6.);
	float ct2=cos(M_PI/12.);
	int dimx=nx+2*ne;
	int dimy=ny+2*ne;
	int dimz=nz+2*ne;
	int dimxy=dimx*dimy;
	int dimxyz=dimxy*dimz;
	
	int srcx=nx/2;
	int srcy=ny/2;
	int srcz=1;

	// allocate
	w = (float*)malloc(sizeof(float)*nt);
	um = (float*)malloc(sizeof(float)*dimxyz);
	uo = (float*)malloc(sizeof(float)*dimxyz); 
	up = (float*)malloc(sizeof(float)*dimxyz); 
	vel = (float*)malloc(sizeof(float)*dimxyz); 
    seismo=(float*)malloc(sizeof(float)*(nt*nx));
	// source wavelet
	FILE *fp = fopen("wavelet.txt","rt");
	for(it=0;it<nt;it++)
		fscanf(fp,"%f",&w[it]);
	fclose(fp);
	
	for(i=0;i<nx*nt;i++)
		seismo[i]=0.;
	for(i=0;i<dimxyz;i++)
	{
		up[i]=0.;
		uo[i]=0.;
		um[i]=0.;
	}
	FILE *fv=fopen("R_SEG_C3NA_Velocity.km.bin","rb");
	fread(vel,sizeof(float),dimxyz,fv);
	
	printf("nt, dt, h = %d %f %f\n", nt, dt, h);
	printf("nx, ny, nz = %d %d %d\n", nx, ny, nz);
	printf("order, ne = %d %d\n", order, ne);

	fp=fopen("20m_c_3dsalt.bin", "wb");
	printf("start\n");
	clock_t start = clock(), diff;
	time3d(seismo,up,uo,um,tmp,vel,nx,ny,nz,ne,dt,h,w,srcx,srcy,srcz,nt,dimx,dimy,dimz,dimxy,fp);
	diff = clock() - start;
	int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("loop end, t=%d ms\n", msec);
	
	fwrite((void*)seismo,sizeof(float),nx*nt,fp);
	fclose(fp);
	fclose(fv);
	//deallocate
	free(w);
	free(up);
	free(uo);
	free(um);
}
