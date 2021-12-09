//cuda loop-plaquette code
#include <iostream>
#include <math.h>
#include <random>
#include <time.h>
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <string.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include "parameters.h"

using namespace std::chrono;
using namespace std;

__host__ __device__ int mod(int a,int b){
    return (a%b + b)%b;
}

__device__ void shiftx(int b[], int a[], int sign, int v){
    for (int i=0; i<d; i++){
        if (i==v){
            if (v==0){ b[i]=mod(a[i]+sign,Nt);  }
            else { b[i]=mod(a[i]+sign,Ns);  }
        }
        else { b[i]=a[i]; }
    }
}

double I(int s){
    double a=0,r=0;
    while(r<inf){
        a+=dr*pow(r,s+1)*exp(-eta*pow(r,2)-lmd*pow(r,4));
        r+=dr;
    }
    return a;
}

__host__ __device__ int sx(int x[], int k[], int a[]){
    int sum=0;
    int v[d]={0};
    for (int i=0;i<d;i++){
        v[i]=1;
        
        sum+=abs(k[x[0]+Nt*x[1]+Nt*Ns*i])
        +abs(k[mod(x[0]-v[0],Nt)+Nt*mod(x[1]-v[1],Ns)+Nt*Ns*i])
        +2*(a[x[0]+Nt*x[1]+Nt*Ns*i]
        +a[mod(x[0]-v[0],Nt)+Nt*mod(x[1]-v[1],Ns)+Nt*Ns*i]);

        v[i]=0;
    }
    return sum;
}
__device__ int countl(int *l){
    int sum=0;
    for (int i=0; i<Nt; i++){
        for (int j=0; j<Ns; j++){
            for (int m=0; m<d; m++) sum+=abs(l[i+j*Nt+m*Nt*Ns]);
        }
    }
    return sum;
}
__global__ void a_update(int t, int tag, int *k, int *a, int *a_, double *I_val){
    int y, x[d], x_[d];
    double rho;
    int id=threadIdx.x + blockDim.x * blockIdx.x;
    
    if (id>Ns*Nt/2-1) return;
    if (t==0){
        x[0]=(2*id+tag)%Nt;
        x[1]=(2*id+tag)/Nt;
    }
    else {
        x[0]=(2*id+tag)/Ns;
        x[1]=(2*id+tag)%Ns;
    }
    
    curandState state;
    curand_init((unsigned long long)clock() + id, 0, 0, &state);
    double rand1 = curand_uniform_double(&state);
    int rand2 = 2*((int)(2*(1-curand_uniform_double(&state))))-1;
    
    y=a_[x[0]+Nt*x[1]+Nt*Ns*t];
    a_[x[0]+Nt*x[1]+Nt*Ns*t]=a[x[0]+Nt*x[1]+Nt*Ns*t]+rand2;
    
    if (a_[x[0]+Nt*x[1]+Nt*Ns*t]<0){
        a_[x[0]+Nt*x[1]+Nt*Ns*t]=y;
        return;
    }
    
    shiftx(x_, x, 1, t);
    if (a_[x[0]+Nt*x[1]+Nt*Ns*t]>a[x[0]+Nt*x[1]+Nt*Ns*t]){
        rho=1.0/(abs(k[x[0]+Nt*x[1]+Nt*Ns*t])+a_[x[0]+Nt*x[1]+Nt*Ns*t])
        /a_[x[0]+Nt*x[1]+Nt*Ns*t]
        *I_val[sx(x,k,a_)]*I_val[sx(x_,k,a_)]
        /I_val[sx(x,k,a)]/I_val[sx(x_,k,a)];
    } 
    else{
        rho=1.0*(abs(k[x[0]+Nt*x[1]+Nt*Ns*t])+a[x[0]+Nt*x[1]+Nt*Ns*t])
        *a[x[0]+Nt*x[1]+Nt*Ns*t]
        *I_val[sx(x,k,a_)]*I_val[sx(x_,k,a_)]
        /I_val[sx(x,k,a)]/I_val[sx(x_,k,a)];
    }
    
    if (rand1<rho){
        a[x[0]+Nt*x[1]+Nt*Ns*t]=a_[x[0]+Nt*x[1]+Nt*Ns*t];
    }
    else{
        a_[x[0]+Nt*x[1]+Nt*Ns*t]=y;
    }
    
}
__global__ void plaquette_update(int i, int j, int *k, int *a, double *I_val){
    int x[d], x1[d], x2[d], x12[d], del;
    double rho=1.0;
    
    int id=threadIdx.x + blockDim.x * blockIdx.x;
    if (id>Ns*Nt/4-1) return;
    x[0]=(2*id+i)%Nt;
    x[1]=((2*id+i)/Nt)*2+j;
    
    curandState state;
    curand_init((unsigned long long)clock() + id, 0, 0, &state);
    double rand1 = curand_uniform_double(&state);
    int rand2 = 2*((int)(2*(1-curand_uniform_double(&state))))-1;
    
    del=rand2;
    
    shiftx(x2,x,1,1);
    shiftx(x1,x,1,0);
    shiftx(x12,x1,1,1);
    
    if (del>0)
    {
        if (k[x[0]+Nt*x[1]+Nt*Ns]>=0) rho*=1.0/(k[x[0]+Nt*x[1]+Nt*Ns]+1+a[x[0]+Nt*x[1]+Nt*Ns]);
        else rho*=abs(k[x[0]+Nt*x[1]+Nt*Ns])+a[x[0]+Nt*x[1]+Nt*Ns];
        if (k[x2[0]+Nt*x2[1]]>=0) rho*=1.0/(k[x2[0]+Nt*x2[1]]+1+a[x2[0]+Nt*x2[1]]);
        else rho*=abs(k[x2[0]+Nt*x2[1]])+a[x2[0]+Nt*x2[1]];
        if (k[x[0]+Nt*x[1]]>0) rho*=k[x[0]+Nt*x[1]]+a[x[0]+Nt*x[1]];
        else rho*=1.0/(abs(k[x[0]+Nt*x[1]])+1+a[x[0]+Nt*x[1]]);
        if (k[x1[0]+Nt*x1[1]+Nt*Ns]>0) rho*=k[x1[0]+Nt*x1[1]+Nt*Ns]+a[x1[0]+Nt*x1[1]+Nt*Ns];
        else rho*=1.0/(abs(k[x1[0]+Nt*x1[1]+Nt*Ns])+1+a[x1[0]+Nt*x1[1]+Nt*Ns]);
    
        if (k[x[0]+Nt*x[1]+Nt*Ns]>=0 && k[x[0]+Nt*x[1]]<=0) rho*=I_val[sx(x,k,a)+2]/I_val[sx(x,k,a)];
        else if (k[x[0]+Nt*x[1]+Nt*Ns]<0 && k[x[0]+Nt*x[1]]>0) rho*=I_val[sx(x,k,a)-2]/I_val[sx(x,k,a)];
        if (k[x2[0]+Nt*x2[1]]>=0 && k[x1[0]+Nt*x1[1]+Nt*Ns]<=0) rho*=I_val[sx(x12,k,a)+2]/I_val[sx(x12,k,a)];
        else if (k[x2[0]+Nt*x2[1]]<0 && k[x1[0]+Nt*x1[1]+Nt*Ns]>0) rho*=I_val[sx(x12,k,a)-2]/I_val[sx(x12,k,a)];
        if (k[x2[0]+Nt*x2[1]]>=0 && k[x[0]+Nt*x[1]+Nt*Ns]>=0) rho*=I_val[sx(x2,k,a)+2]/I_val[sx(x2,k,a)];
        else if (k[x2[0]+Nt*x2[1]]<0 && k[x[0]+Nt*x[1]+Nt*Ns]<0) rho*=I_val[sx(x2,k,a)-2]/I_val[sx(x2,k,a)];
        if (k[x1[0]+Nt*x1[1]+Nt*Ns]<=0 && k[x[0]+Nt*x[1]]<=0) rho*=I_val[sx(x1,k,a)+2]/I_val[sx(x1,k,a)];
        else if(k[x1[0]+Nt*x1[1]+Nt*Ns]>0 && k[x[0]+Nt*x[1]]>0) rho*=I_val[sx(x1,k,a)-2]/I_val[sx(x1,k,a)];
    } 
    else
    {
        if (k[x[0]+Nt*x[1]+Nt*Ns]<=0) rho*=1.0/(abs(k[x[0]+Nt*x[1]+Nt*Ns])+1+a[x[0]+Nt*x[1]+Nt*Ns]);
        else rho*=k[x[0]+Nt*x[1]+Nt*Ns]+a[x[0]+Nt*x[1]+Nt*Ns];
        if (k[x2[0]+Nt*x2[1]]<=0) rho*=1.0/(abs(k[x2[0]+Nt*x2[1]])+1+a[x2[0]+Nt*x2[1]]);
        else rho*=k[x2[0]+Nt*x2[1]]+a[x2[0]+Nt*x2[1]];
        if (k[x[0]+Nt*x[1]]<0)rho*=abs(k[x[0]+Nt*x[1]])+a[x[0]+Nt*x[1]];
        else rho*=1.0/(k[x[0]+Nt*x[1]]+1+a[x[0]+Nt*x[1]]);
        if (k[x1[0]+Nt*x1[1]+Nt*Ns]<0) rho*=abs(k[x1[0]+Nt*x1[1]+Nt*Ns])+a[x1[0]+Nt*x1[1]+Nt*Ns];
        else rho*=1.0/(k[x1[0]+Nt*x1[1]+Nt*Ns]+1+a[x1[0]+Nt*x1[1]+Nt*Ns]);
    
        if (k[x[0]+Nt*x[1]+Nt*Ns]<=0 && k[x[0]+Nt*x[1]]>=0) rho*=I_val[sx(x,k,a)+2]/I_val[sx(x,k,a)];
        else if (k[x[0]+Nt*x[1]+Nt*Ns]>0 && k[x[0]+Nt*x[1]]<0) rho*=I_val[sx(x,k,a)-2]/I_val[sx(x,k,a)];
        if (k[x2[0]+Nt*x2[1]]<=0 && k[x1[0]+Nt*x1[1]+Nt*Ns]>=0) rho*=I_val[sx(x12,k,a)+2]/I_val[sx(x12,k,a)];
        else if (k[x2[0]+Nt*x2[1]]>0 && k[x1[0]+Nt*x1[1]+Nt*Ns]<0) rho*=I_val[sx(x12,k,a)-2]/I_val[sx(x12,k,a)];
        if (k[x2[0]+Nt*x2[1]]<=0 && k[x[0]+Nt*x[1]+Nt*Ns]<=0) rho*=I_val[sx(x2,k,a)+2]/I_val[sx(x2,k,a)];
        else if (k[x2[0]+Nt*x2[1]]>0 && k[x[0]+Nt*x[1]+Nt*Ns]>0) rho*=I_val[sx(x2,k,a)-2]/I_val[sx(x2,k,a)];
        if (k[x1[0]+Nt*x1[1]+Nt*Ns]>=0 && k[x[0]+Nt*x[1]]>=0) rho*=I_val[sx(x1,k,a)+2]/I_val[sx(x1,k,a)];
        else if(k[x1[0]+Nt*x1[1]+Nt*Ns]<0 && k[x[0]+Nt*x[1]]<0) rho*=I_val[sx(x1,k,a)-2]/I_val[sx(x1,k,a)];
    }
    if (rand1<rho)
    {
        k[x[0]+Nt*x[1]+Nt*Ns]+=del;
        k[x2[0]+Nt*x2[1]]+=del;
        k[x1[0]+Nt*x1[1]+Nt*Ns]-=del;
        k[x[0]+Nt*x[1]]-=del;
    }
}
__global__ void delrandvals(int *delrand, int N){
    
    int id=threadIdx.x + blockDim.x * blockIdx.x;
    if (id>N-1) return;
    curandState state;
    curand_init((unsigned long long)clock() + id, 0, 0, &state);
    delrand[id] = 2*((int)(2*(1-curand_uniform_double(&state))))-1;
}
__global__ void k_rho(int t, int *delrand, double *rho, int *k, int *a, double mu, double *I_val) {
    
    int x[d], x_[d];
    int id=threadIdx.x + blockDim.x * blockIdx.x;
    if (id>Ns*Nt-1) return;
    x[0]=id%Nt;
    x[1]=id/Nt;
    rho[x[0]+Nt*x[1]]=1.0;
    int del=delrand[x[1-t]];
    shiftx(x_,x,-1,t);
    
    if (del>0)
    {
        if (k[x[0]+Nt*x[1]+Nt*Ns*t]>=0) rho[x[0]+Nt*x[1]]*=1.0/(k[x[0]+Nt*x[1]+Nt*Ns*t]+1+a[x[0]+Nt*x[1]+Nt*Ns*t]);
        else rho[x[0]+Nt*x[1]]*=abs(k[x[0]+Nt*x[1]+Nt*Ns*t])+a[x[0]+Nt*x[1]+Nt*Ns*t];
        if (k[x[0]+Nt*x[1]+Nt*Ns*t]>=0 && k[x_[0]+Nt*x_[1]+Nt*Ns*t]>=0) rho[x[0]+Nt*x[1]]*=I_val[sx(x,k,a)+2]/I_val[sx(x,k,a)];
        else if (k[x[0]+Nt*x[1]+Nt*Ns*t]<0 && k[x_[0]+Nt*x_[1]+Nt*Ns*t]<0) rho[x[0]+Nt*x[1]]*=I_val[sx(x,k,a)-2]/I_val[sx(x,k,a)];
    }
    else
    {
        if (k[x[0]+Nt*x[1]+Nt*Ns*t]>0) rho[x[0]+Nt*x[1]]*=k[x[0]+Nt*x[1]+Nt*Ns*t]+a[x[0]+Nt*x[1]+Nt*Ns*t];
        else rho[x[0]+Nt*x[1]]*=1.0/(abs(k[x[0]+Nt*x[1]+Nt*Ns*t])+a[x[0]+Nt*x[1]+Nt*Ns*t]+1);
        if (k[x[0]+Nt*x[1]+Nt*Ns*t]<=0 && k[x_[0]+Nt*x_[1]+Nt*Ns*t]<=0) rho[x[0]+Nt*x[1]]*=I_val[sx(x,k,a)+2]/I_val[sx(x,k,a)];
        else if (k[x[0]+Nt*x[1]+Nt*Ns*t]>0 && k[x_[0]+Nt*x_[1]+Nt*Ns*t]>0) rho[x[0]+Nt*x[1]]*=I_val[sx(x,k,a)-2]/I_val[sx(x,k,a)];
    }
    rho[id]*=exp(del*mu*(t==0));
}
__global__ void temporalloop_update(double *rho_vals, int *rand2, int *k){
    
    double rho=1.0;
    int id=threadIdx.x + blockDim.x * blockIdx.x;
    if (id>Ns-1) return;
    
    curandState state;
    curand_init((unsigned long long)clock() + id, 0, 0, &state);
    double rand1 = curand_uniform_double(&state);
    
    for (int bx=0; bx<Nt; bx++){
        rho*=rho_vals[bx+Nt*id];
    }
    if (rand1<rho){
        for (int bx=0; bx<Nt; bx++){
            k[bx+Nt*id]+=rand2[id];
        }
    }
    
}
__global__ void spatialloop_update(double *rho_vals, int *rand2, int *k){
    
    double rho=1.0;
    int id=threadIdx.x + blockDim.x * blockIdx.x;
    if (id>Nt-1) return;
    
    curandState state;
    curand_init((unsigned long long)clock() + id, 0, 0, &state);
    double rand1 = curand_uniform_double(&state);
    
    for (int by=0; by<Ns; by++){
        rho*=rho_vals[id+Nt*by];
    }
    if (rand1<rho){
        for (int by=0; by<Ns; by++){
            k[id+Nt*by+Nt*Ns]+=rand2[id];
        }
    }
}

void update(int *k, int *a, int *a_, double mu, double *I_val, bool flag2){
    
    blocks=Ns*Nt/2/blockSize+1;
    
    a_update<<<blocks,blockSize>>>(0,0,k,a,a_,I_val);
    a_update<<<blocks,blockSize>>>(0,1,k,a,a_,I_val);
    a_update<<<blocks,blockSize>>>(1,0,k,a,a_,I_val);
    a_update<<<blocks,blockSize>>>(1,1,k,a,a_,I_val);
    
    blocks=Ns*Nt/4/blockSize+1;
    plaquette_update<<<blocks,blockSize>>>(0,0,k,a,I_val);
    plaquette_update<<<blocks,blockSize>>>(0,1,k,a,I_val);
    plaquette_update<<<blocks,blockSize>>>(1,0,k,a,I_val);
    plaquette_update<<<blocks,blockSize>>>(1,1,k,a,I_val);
    
    int *delrand;
    double *rho;

    cudaMalloc(&rho, Nt*Ns*sizeof(*rho));
    cudaMalloc(&delrand, Ns*sizeof(*delrand));
	
    blocks=Ns/blockSize+1;
    delrandvals<<<blocks,blockSize>>>(delrand,Ns);
    
    blocks=Ns*Nt/blockSize+1;
    k_rho<<<blocks,blockSize>>>(0,delrand,rho,k,a,mu,I_val);
    
    blocks=Ns/blockSize+1;
    temporalloop_update<<<blocks,blockSize>>>(rho,delrand,k);
    cudaFree(delrand);
    
    cudaMalloc(&delrand, Nt*sizeof(*delrand));
    
    blocks=Nt/blockSize+1;
    delrandvals<<<blocks,blockSize>>>(delrand,Nt);
    
    blocks=Ns*Nt/blockSize+1;
    k_rho<<<blocks,blockSize>>>(1,delrand,rho,k,a,mu,I_val);
    
    blocks=Nt/blockSize+1;
    spatialloop_update<<<blocks,blockSize>>>(rho,delrand,k);
    
}

__global__ void init_lattice(int *A, int a, int n){
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i<n) A[i]=a;
}


__global__ void arraycopy(int *A, int *B, int n){
    int id=threadIdx.x+blockDim.x*blockIdx.x;
    if (id<n) A[id]=B[id];
}
int ksum(int k[]){
    int sum=0;
    for (int i=0;i<Nt;i++){
        for (int j=0;j<Ns;j++){
            sum+=k[i+j*Nt];
        }
    }
    return sum;
}

double phi2(int k[], int a[], double I[int_val]){
    double sum=0;
    int x[d];
    for (int i=0;i<Nt;i++){
        for (int j=0;j<Ns;j++){
            x[0]=i; x[1]=j;
            sum+=I[sx(x, k, a)+2]/I[sx(x, k, a)];
        }
    }
    return sum/Nt/Ns;
}
double phi4(int k[], int a[], double I[int_val]){
    double sum=0;
    int x[d];
    for (int i=0;i<Nt;i++){
        for (int j=0;j<Ns;j++){
            x[0]=i; x[1]=j;
            sum+=I[sx(x, k, a)+4]/I[sx(x, k, a)];
        }
    
    }
    return sum/Nt/Ns;
}

double errorjack(double *xi, int configs){
    double *x_i, x_=0, stddev=0;
    x_i=(double*) malloc(configs*sizeof(*x_i));
    for (int i=0; i<configs; i++){
        x_i[i]=0;
        for (int j=0; j<configs; j++){
            x_i[i]+=(1-(i==j))*xi[j];
        }
        x_i[i]=x_i[i]/(configs-1);
        x_+=x_i[i];
    }
    x_=x_/configs;
    for (int i=0; i<configs; i++){
        stddev+=(x_i[i]-x_)*(x_i[i]-x_);
    }
    stddev=sqrt(stddev*(configs-1)/configs);
    return stddev;
}

int main(int argc, char **argv)
{
    int *k, *a, *a_, *kh, *ah, *a_h;
    double *I_val;
    
    cudaMalloc(&k, Nt*Ns*d*sizeof(*k));
    cudaMalloc(&a, Nt*Ns*d*sizeof(*a));
    cudaMalloc(&a_, Nt*Ns*d*sizeof(*a_));
    cudaMalloc(&I_val, int_val*sizeof(*I_val));
    
    kh=(int*) malloc(Nt*Ns*d*sizeof(*kh));
    ah=(int*) malloc(Nt*Ns*d*sizeof(*ah));
    a_h=(int*) malloc(Nt*Ns*d*sizeof(*a_h));
    
    double dmu=(mu_max-mu_min)/mu_n; 
    double n_avg, phi2_avg, phi4_avg;
    double *xi,*phi2i, I_val_h[int_val];
    double mu=mu_max-dmu;
    
    xi=(double*) malloc(configs*sizeof(*xi));
    phi2i=(double*) malloc(configs*sizeof(*phi2i));
    
    for (int i=0; i<int_val; i++){
        I_val_h[i]=I(i);
    }
    cudaMemcpy(I_val, I_val_h, int_val*sizeof(*I_val), cudaMemcpyHostToDevice);
    ofstream data, data1, data2, data4, data5;
    
	string filename1="mu_vs_n.txt"
    ,filename2="mu_vs_phi2.txt";
	
    data1.open(filename1);
    data2.open(filename2);
    
    blocks=Nt*Ns*d/blockSize+1;
    
    auto begin=high_resolution_clock::now();
    
    for (int g=0; g<mu_n; g++){
	
		init_lattice<<<blocks,blockSize>>>(k,0,Nt*Ns*d);
		init_lattice<<<blocks,blockSize>>>(a,0,Nt*Ns*d);
		init_lattice<<<blocks,blockSize>>>(a_,0,Nt*Ns*d);
        double mu_therm=1.3;
        for (int i=0; i<equil; i++){
            update(k,a,a_,mu_therm,I_val,false); 
        }
		
		for (int i=0; i<equil; i++){
            update(k,a,a_,mu,I_val,false); 
        }
        phi2_avg=0;
        n_avg=0;
        
        for (int i=0; i<configs; i++){
            for (int j=0; j<gaps; j++){
                update(k,a,a_,mu,I_val,true);
            }
            update(k,a,a_,mu,I_val,true);
            cudaDeviceSynchronize();
            
            cudaMemcpy(kh, k, Nt*Ns*d*sizeof(*k), cudaMemcpyDeviceToHost);
            cudaMemcpy(ah, a, Nt*Ns*d*sizeof(*a), cudaMemcpyDeviceToHost);
            
			xi[i]=1.0*ksum(kh)/Nt/Ns;
            phi2i[i]=phi2(kh,ah,I_val_h);
            
            n_avg+=xi[i];
            phi2_avg+=phi2i[i];
        }
        n_avg=n_avg/configs;
        phi2_avg=phi2_avg/configs;
        
        data1<<mu<<"\t"<<n_avg<<"\t"<<errorjack(xi,configs)<<"\n";
        data2<<mu<<"\t"<<phi2_avg<<"\t"<<errorjack(phi2i,configs)<<"\n";
        mu-=dmu;
        cout<<g<<endl;
	
    } 
    
    data1.close();
    data2.close();
    cudaFree(k);
    cudaFree(a);
    cudaFree(a_);
    cudaFree(I_val);
    free(kh);
    free(ah);
    free(a_h);
    
    auto stop=high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-begin);
    
    cout<<duration.count()<<endl;
    
    return 0;
}
