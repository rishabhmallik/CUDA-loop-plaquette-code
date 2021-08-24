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
//#include "observables.h"

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
__global__ void a_update(int t, int tag, int *k, int *a, int *a_, double *I_val){
    int y, x[d], x_[d];
    double rho;
    int id=threadIdx.x + blockDim.x * blockIdx.x;
    //printf("a_begin\n");
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
        //printf("ar\n");
        a[x[0]+Nt*x[1]+Nt*Ns*t]=a_[x[0]+Nt*x[1]+Nt*Ns*t];
    }
    else{
        a_[x[0]+Nt*x[1]+Nt*Ns*t]=y;
    }
    //printf("a\n");
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
    shiftx(x_,x,-1,1);
    
    if (del>0)
    {
        if (k[x[0]+Nt*x[1]+Nt*Ns*t]>=0) rho[x[0]+Nt*x[1]]*=1.0/(k[x[0]+Nt*x[1]+Nt*Ns*t]+1+a[x[0]+Nt*x[1]+Nt*Ns*t]);
        else rho[x[0]+Nt*x[1]]*=abs(k[x[0]+Nt*x[1]+Nt*Ns*t])+a[x[0]+Nt*x[1]+Nt*Ns*t];
        if (k[x[0]+Nt*x[1]+Nt*Ns*t]>=0 && k[x_[0]+Nt*x_[1]+Nt*Ns*t]>=0) rho[x[0]+Nt*x[1]]*=exp(del*mu*(t==0))*I_val[sx(x,k,a)+2]/I_val[sx(x,k,a)];
        else if (k[x[0]+Nt*x[1]+Nt*Ns*t]<0 && k[x_[0]+Nt*x_[1]+Nt*Ns*t]<0) rho[x[0]+Nt*x[1]]*=exp(del*mu*(t==0))*I_val[sx(x,k,a)-2]/I_val[sx(x,k,a)];
    }
    else
    {
        if (k[x[0]+Nt*x[1]+Nt*Ns*t]>0) rho[x[0]+Nt*x[1]]*=k[x[0]+Nt*x[1]+Nt*Ns*t]+a[x[0]+Nt*x[1]+Nt*Ns*t];
        else rho[x[0]+Nt*x[1]]*=1.0/(abs(k[x[0]+Nt*x[1]+Nt*Ns*t])+a[x[0]+Nt*x[1]+Nt*Ns*t]+1);
        if (k[x[0]+Nt*x[1]+Nt*Ns*t]<=0 && k[x_[0]+Nt*x_[1]+Nt*Ns*t]<=0) rho[x[0]+Nt*x[1]]*=exp(del*mu*(t==0))*I_val[sx(x,k,a)+2]/I_val[sx(x,k,a)];
        else if (k[x[0]+Nt*x[1]+Nt*Ns*t]>0 && k[x_[0]+Nt*x_[1]+Nt*Ns*t]>0) rho[x[0]+Nt*x[1]]*=exp(del*mu*(t==0))*I_val[sx(x,k,a)-2]/I_val[sx(x,k,a)];
    }
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
    //printf("t%d\n", id);
    
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
    //printf("spat\n");
}

void update(int *k, int *a, int *a_, double mu, double *I_val){
    
    //threadsPerBlock=(Ns*Nt/2-1)%blockSize+1;
    blocks=Ns*Nt/2/blockSize+1;
    //printf("before a\n");
    a_update<<<blocks,blockSize>>>(0,0,k,a,a_,I_val);
    a_update<<<blocks,blockSize>>>(0,1,k,a,a_,I_val);
    a_update<<<blocks,blockSize>>>(1,0,k,a,a_,I_val);
    a_update<<<blocks,blockSize>>>(1,1,k,a,a_,I_val);
    
    //threadsPerBlock=(Ns*Nt/4-1)%blockSize+1;
    blocks=Ns*Nt/4/blockSize+1;
    plaquette_update<<<blocks,blockSize>>>(0,0,k,a,I_val);
    plaquette_update<<<blocks,blockSize>>>(0,1,k,a,I_val);
    plaquette_update<<<blocks,blockSize>>>(1,0,k,a,I_val);
    plaquette_update<<<blocks,blockSize>>>(1,1,k,a,I_val);
    
    int *delrand;
    double *rho;

    cudaMalloc(&rho, Nt*Ns*sizeof(*rho));
    cudaMalloc(&delrand, Ns*sizeof(*delrand));

    //threadsPerBlock=(Ns-1)%blockSize+1;
    blocks=Ns/blockSize+1;
    delrandvals<<<blocks,blockSize>>>(delrand,Ns);
    
    //threadsPerBlock=(Ns*Nt-1)%blockSize+1;
    blocks=Ns*Nt/blockSize+1;
    k_rho<<<blocks,blockSize>>>(0,delrand,rho,k,a,mu,I_val);
    
    //threadsPerBlock=(Ns-1)%blockSize+1;
    blocks=Ns/blockSize+1;
    temporalloop_update<<<blocks,blockSize>>>(rho,delrand,k);
    cudaFree(delrand);
    
    cudaMalloc(&delrand, Nt*sizeof(*delrand));
    //threadsPerBlock=(Nt-1)%blockSize+1;
    blocks=Nt/blockSize+1;
    delrandvals<<<blocks,blockSize>>>(delrand,Nt);
    
    //threadsPerBlock=(Ns*Nt-1)%blockSize+1;
    blocks=Ns*Nt/blockSize+1;
    k_rho<<<blocks,blockSize>>>(1,delrand,rho,k,a,mu,I_val);
    
    //threadsPerBlock=(Nt-1)%blockSize+1;
    blocks=Nt/blockSize+1;
    spatialloop_update<<<blocks,blockSize>>>(rho,delrand,k);
    
}

__global__ void init_lattice(int *A){
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    A[i]=0;
}


__global__ void arraycopy(int *A, int *B, int n){
    int id=threadIdx.x+blockDim.x*blockIdx.x;
    if (id<n) A[id]=B[id];
}
__global__ void sumreduceint(int *sum, int p, int t, int n){
    int id=threadIdx.x+blockDim.x*blockIdx.x;

    if(id%(t*p)==0 && id<n){
        for (int i=id+p; (i<id+t*p && i<n); i+=p){
            sum[id]+=sum[i];
        }
    }
}

int ksum(int *k){
    int n=Nt*Ns;
    int *sum, *sumh;
    int *kcpy;
    cudaMalloc(&kcpy, Nt*Ns*d*sizeof(*kcpy));
    sumh=(int*) malloc (sizeof(*sumh));
    cudaMalloc(&sum, sizeof(*sum));
    arraycopy<<<n/blockSize+1, blockSize>>>(kcpy, k, n);
    int p=1;
    do{
        sumreduceint<<<n/blockSize+1, blockSize>>>(kcpy, p, t, n);
        cudaDeviceSynchronize();
        p*=d;
    }while(n/p>=1);
    
    sum=kcpy;
    cudaMemcpy(sumh, sum, sizeof(*sum), cudaMemcpyDeviceToHost);
    cudaFree(sum);
    cudaFree(kcpy);
    return *sumh;
}

__global__ void sumreducedouble(double *sum, int p, int t, int n){
    int id=threadIdx.x+blockDim.x*blockIdx.x;

    if(id%(t*p)==0 && id<n){
        for (int i=id+p; (i<id+t*p && i<n); i+=p){
            sum[id]+=sum[i];
        }
    }
}
__global__ void Iratio2(double *I, double *r, int *k, int *a, int n){
    int id=threadIdx.x + blockDim.x * blockIdx.x;
    int x[d];
    x[0]=id%Nt;
    x[1]=id/Nt;
    if (id<n){
        r[id]=I[sx(x, k, a)+2]/I[sx(x, k, a)];
    }
}

float phi2(int *k, int *a, double *I){
    int n=Nt*Ns;
    double *sum, *sumh;
    double *r;
    cudaMalloc(&r, Nt*Ns*sizeof(*r));
    sumh=(double*) malloc (sizeof(*sumh));
    cudaMalloc(&sum, sizeof(*sum));
    Iratio2<<<n/blockSize+1, blockSize>>>(I, r, k, a, n);
    int p=1;
    do{
        sumreducedouble<<<n/blockSize+1, blockSize>>>(r, p, t, n);
        cudaDeviceSynchronize();
        p*=t;
    }while(n/p>=1);
    
    sum=r;
    cudaMemcpy(sumh, sum, sizeof(*sum), cudaMemcpyDeviceToHost);
    cudaFree(r);
    cudaFree(sum);
    return *sumh/Ns/Nt;
}

float errorjack(double xi[configs]){
    double x_i[configs], x_=0, stddev=0;
    for (int i=0; i<configs; i++){
        for (int j=0; j<configs; j++){
            x_i[i]=(1-(i==j))*xi[j];
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
    auto begin=high_resolution_clock::now();
    
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
    double xi[configs],phi2i[configs], I_val_h[int_val];
    double mu=mu_min;
    
    for (int i=0; i<int_val; i++){
        I_val_h[i]=I(i);
    }
    cudaMemcpy(I_val, I_val_h, int_val*sizeof(*I_val), cudaMemcpyHostToDevice);
    ofstream data, data1;
    /*
    string filename="mu_n_phi2_phi4_Nt"+to_string(Nt)+"_Ns"+to_string(Ns)+"_eta"
        +to_string(eta)+"_lmd"+to_string(lmd)+"_"+to_string(configs)+"_"+to_string(gaps)+"_"
        +to_string(equil)+".txt";
    */
    data.open("mu_vs_n.txt");
    data1.open("mu_vs_phi2.txt");
    //data2.open("mu_vs_phi4.txt");
    
    //dim3 threadblock = dim3(Nt, Ns);
    blocks=Nt*Ns*d/blockSize+1;
    //threadsPerBlock=(Nt*Ns*d-1)%blockSize+1;
    init_lattice<<<blocks,blockSize>>>(k);
    init_lattice<<<blocks,blockSize>>>(a);
    init_lattice<<<blocks,blockSize>>>(a_);
    
    for (int g=0; g<mu_n; g++){
        
        for (int i=0; i<equil; i++){
            update(k,a,a_,mu,I_val);                                                                   }
        cudaDeviceSynchronize();
        phi2_avg=0;
        n_avg=0;
        //phi4_avg=0;
        for (int i=0; i<configs; i++){
            for (int j=0; j<gaps; j++){
                update(k,a,a_,mu,I_val);
		//cudaDeviceSynchronize();
            }
            update(k,a,a_,mu,I_val);
            cudaDeviceSynchronize();
            
            cudaMemcpy(kh, k, Nt*Ns*d*sizeof(*k), cudaMemcpyDeviceToHost);
            cudaMemcpy(ah, a, Nt*Ns*d*sizeof(*a), cudaMemcpyDeviceToHost);
            cudaMemcpy(a_h, a_, Nt*Ns*d*sizeof(*a_), cudaMemcpyDeviceToHost);
            
            xi[i]=1.0*ksum(kh)/Nt/Ns;
            phi2i[i]=phi2(kh,a,I_val);
            
            n_avg+=xi[i];
            phi2_avg+=phi2i[i];
            //phi4_avg+=phi4(k,a,I_val);
        }
        n_avg=n_avg/configs;
        phi2_avg=phi2_avg/configs;
        //phi4_avg=phi4_avg/configs;
        
        data<<mu<<"\t"<<n_avg<<"\t"<<errorjack(xi)<<"\n";
        data1<<mu<<"\t"<<phi2_avg<<"\t"<<errorjack(phi2i)<<"\n";
        mu+=dmu;
        cout<<g<<endl;
        
    }
    
    data.close();
    data1.close();
    //data2.close();
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
