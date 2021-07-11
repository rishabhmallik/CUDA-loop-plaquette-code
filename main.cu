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

int ksum(int k[][Ns][d]){
    int sum=0;
    for (int i=0;i<Nt;i++){
        for (int j=0;j<Ns;j++){
            sum+=k[i][j][0];
        }
    }
    return sum;
}
__host__ __device__ int sx(int x[], int k[][Ns][d], int a[][Ns][d]){
    int sum=0;
    int v[d]={0};
    for (int i=0;i<d;i++){
        v[i]=1;

        sum+=abs(k[x[0]][x[1]][i])
        +abs(k[mod(x[0]-v[0],Nt)][mod(x[1]-v[1],Ns)][i])
        +2*(a[x[0]][x[1]][i]
        +a[mod(x[0]-v[0],Nt)][mod(x[1]-v[1],Ns)][i]);

        v[i]=0;
    }
    return sum;
}
__device__ void a_update(int t){
    int y, x[d], x_[d];
    double rho;
    
    x[0]=threadIdx.x;
    x[1]=threadIdx.y;
    
    y=a_[x[0]][x[1]][t];
    a_[x[0]][x[1]][t]=a[x[0]][x[1]][t]+rand2[Nt*x[1]+x[0]];
    
    if (a_[x[0]][x[1]][t]<0){
        a_[x[0]][x[1]][t]=y;
        return;
    }
    
    shiftx(x_, x, 1, t);
    if (a_[x[0]][x[1]][t]>a[x[0]][x[1]][t]){
        rho=1.0/(abs(k[x[0]][x[1]][t])+a_[x[0]][x[1]][t])
        /a_[x[0]][x[1]][t]
        *I_val[sx(x,k,a_)]*I_val[sx(x_,k,a_)]
        /I_val[sx(x,k,a)]/I_val[sx(x_,k,a)];
    } 
    else{
        rho=1.0*(abs(k[x[0]][x[1]][t])+a[x[0]][x[1]][t])
        *a[x[0]][x[1]][t]
        *I_val[sx(x,k,a_)]*I_val[sx(x_,k,a_)]
        /I_val[sx(x,k,a)]/I_val[sx(x_,k,a)];
    }
    if (rand1[Nt*x[1]+x[0]]<rho){
        //printf("ar\n");
        a[x[0]][x[1]][t]=a_[x[0]][x[1]][t];
    }
    else{
        a_[x[0]][x[1]][t]=y;
    }
}
__device__ void plaquette_update(){
    int x[d], x1[d], x2[d], x12[d], del;
    double rho=1.0;
    
    x[0]=threadIdx.x;
    x[1]=threadIdx.y;
    del=rand2[Nt*x[1]+x[0]];
    
    shiftx(x2,x,1,1);
    shiftx(x1,x,1,0);
    shiftx(x12,x1,1,1);
    
    if (del>0)
    {
        if (k[x[0]][x[1]][1]>=0) rho*=1.0/(k[x[0]][x[1]][1]+1+a[x[0]][x[1]][1]);
        else rho*=abs(k[x[0]][x[1]][1])+a[x[0]][x[1]][1];
        if (k[x2[0]][x2[1]][0]>=0) rho*=1.0/(k[x2[0]][x2[1]][0]+1+a[x2[0]][x2[1]][0]);
        else rho*=abs(k[x2[0]][x2[1]][0])+a[x2[0]][x2[1]][0];
        if (k[x[0]][x[1]][0]>0) rho*=k[x[0]][x[1]][0]+a[x[0]][x[1]][0];
        else rho*=1.0/(abs(k[x[0]][x[1]][0])+1+a[x[0]][x[1]][0]);
        if (k[x1[0]][x1[1]][1]>0) rho*=k[x1[0]][x1[1]][1]+a[x1[0]][x1[1]][1];
        else rho*=1.0/(abs(k[x1[0]][x1[1]][1])+1+a[x1[0]][x1[1]][1]);
    
        if (k[x[0]][x[1]][1]>=0 && k[x[0]][x[1]][0]<=0) rho*=I_val[sx(x,k,a)+2]/I_val[sx(x,k,a)];
        else if (k[x[0]][x[1]][1]<0 && k[x[0]][x[1]][0]>0) rho*=I_val[sx(x,k,a)-2]/I_val[sx(x,k,a)];
        if (k[x2[0]][x2[1]][0]>=0 && k[x1[0]][x1[1]][1]<=0) rho*=I_val[sx(x12,k,a)+2]/I_val[sx(x12,k,a)];
        else if (k[x2[0]][x2[1]][0]<0 && k[x1[0]][x1[1]][1]>0) rho*=I_val[sx(x12,k,a)-2]/I_val[sx(x12,k,a)];
        if (k[x2[0]][x2[1]][0]>=0 && k[x[0]][x[1]][1]>=0) rho*=I_val[sx(x2,k,a)+2]/I_val[sx(x2,k,a)];
        else if (k[x2[0]][x2[1]][0]<0 && k[x[0]][x[1]][1]<0) rho*=I_val[sx(x2,k,a)-2]/I_val[sx(x2,k,a)];
        if (k[x1[0]][x1[1]][1]<=0 && k[x[0]][x[1]][0]<=0) rho*=I_val[sx(x1,k,a)+2]/I_val[sx(x1,k,a)];
        else if(k[x1[0]][x1[1]][1]>0 && k[x[0]][x[1]][0]>0) rho*=I_val[sx(x1,k,a)-2]/I_val[sx(x1,k,a)];
    } 
    else
    {
        if (k[x[0]][x[1]][1]<=0) rho*=1.0/(abs(k[x[0]][x[1]][1])+1+a[x[0]][x[1]][1]);
        else rho*=k[x[0]][x[1]][1]+a[x[0]][x[1]][1];
        if (k[x2[0]][x2[1]][0]<=0) rho*=1.0/(abs(k[x2[0]][x2[1]][0])+1+a[x2[0]][x2[1]][0]);
        else rho*=k[x2[0]][x2[1]][0]+a[x2[0]][x2[1]][0];
        if (k[x[0]][x[1]][0]<0)rho*=abs(k[x[0]][x[1]][0])+a[x[0]][x[1]][0];
        else rho*=1.0/(k[x[0]][x[1]][0]+1+a[x[0]][x[1]][0]);
        if (k[x1[0]][x1[1]][1]<0) rho*=abs(k[x1[0]][x1[1]][1])+a[x1[0]][x1[1]][1];
        else rho*=1.0/(k[x1[0]][x1[1]][1]+1+a[x1[0]][x1[1]][1]);
    
        if (k[x[0]][x[1]][1]<=0 && k[x[0]][x[1]][0]>=0) rho*=I_val[sx(x,k,a)+2]/I_val[sx(x,k,a)];
        else if (k[x[0]][x[1]][1]>0 && k[x[0]][x[1]][0]<0) rho*=I_val[sx(x,k,a)-2]/I_val[sx(x,k,a)];
        if (k[x2[0]][x2[1]][0]<=0 && k[x1[0]][x1[1]][1]>=0) rho*=I_val[sx(x12,k,a)+2]/I_val[sx(x12,k,a)];
        else if (k[x2[0]][x2[1]][0]>0 && k[x1[0]][x1[1]][1]<0) rho*=I_val[sx(x12,k,a)-2]/I_val[sx(x12,k,a)];
        if (k[x2[0]][x2[1]][0]<=0 && k[x[0]][x[1]][1]<=0) rho*=I_val[sx(x2,k,a)+2]/I_val[sx(x2,k,a)];
        else if (k[x2[0]][x2[1]][0]>0 && k[x[0]][x[1]][1]>0) rho*=I_val[sx(x2,k,a)-2]/I_val[sx(x2,k,a)];
        if (k[x1[0]][x1[1]][1]>=0 && k[x[0]][x[1]][0]>=0) rho*=I_val[sx(x1,k,a)+2]/I_val[sx(x1,k,a)];
        else if(k[x1[0]][x1[1]][1]<0 && k[x[0]][x[1]][0]<0) rho*=I_val[sx(x1,k,a)-2]/I_val[sx(x1,k,a)];
    }
    if (rand1[Nt*x[1]+x[0]]<rho)
    {
        k[x[0]][x[1]][1]+=del;
        k[x2[0]][x2[1]][0]+=del;
        k[x1[0]][x1[1]][1]-=del;
        k[x[0]][x[1]][0]-=del;
    }
}

__device__ void k_rho(int x[d], int t, int del, double rho[][Ns]) {
    
    rho[x[0]][x[1]]=1.0;
    int x_[d];
    
    shiftx(x_,x,-1,1);
    if (del>0)
    {
        if (k[x[0]][x[1]][t]>=0) rho[x[0]][x[1]]*=1.0/(k[x[0]][x[1]][t]+1+a[x[0]][x[1]][t]);
        else rho[x[0]][x[1]]*=abs(k[x[0]][x[1]][t])+a[x[0]][x[1]][t];
        if (k[x[0]][x[1]][t]>=0 && k[x_[0]][x_[1]][t]>=0) rho[x[0]][x[1]]*=exp(del*mu*(t==0))*I_val[sx(x,k,a)+2]/I_val[sx(x,k,a)];
        else if (k[x[0]][x[1]][t]<0 && k[x_[0]][x_[1]][t]<0) rho[x[0]][x[1]]*=exp(del*mu*(t==0))*I_val[sx(x,k,a)-2]/I_val[sx(x,k,a)];
    } 
    else
    {
        if (k[x[0]][x[1]][t]>0) rho[x[0]][x[1]]*=k[x[0]][x[1]][t]+a[x[0]][x[1]][t];
        else rho[x[0]][x[1]]*=1.0/(abs(k[x[0]][x[1]][t])+a[x[0]][x[1]][t]+1);
        if (k[x[0]][x[1]][t]<=0 && k[x_[0]][x_[1]][t]<=0) rho[x[0]][x[1]]*=exp(del*mu*(t==0))*I_val[sx(x,k,a)+2]/I_val[sx(x,k,a)];
        else if (k[x[0]][x[1]][t]>0 && k[x_[0]][x_[1]][t]>0) rho[x[0]][x[1]]*=exp(del*mu*(t==0))*I_val[sx(x,k,a)-2]/I_val[sx(x,k,a)];
    }
}
__device__ void temporalloop_update(){
    int x[d];
    __shared__ double rho_vals[Nt][Ns];
    
    double rho;
    x[0]=threadIdx.x;
    x[1]=threadIdx.y;
        
    k_rho(x, 0, rand2[x[1]], rho_vals);
    printf("randt=%d\n",rand2[5]);
    for (int by=0; by<Ns; by++){
        rho=1.0;
        for (int bx=0; bx<Nt; bx++){
            rho*=rho_vals[bx][by];
        }
        if (rand1[by]<rho){
            for (int bx=0; bx<Nt; bx++){
                k[bx][by][0]+=rand2[by];
            }
        }
    }
}
__device__ void spatialloop_update(){
    int x[d];
    __shared__ double rho_vals[Nt][Ns];  
    
    double rho;
    x[0]=threadIdx.x;
    x[1]=threadIdx.y;
    
    k_rho(x, 1, rand2[x[0]], rho_vals);
    for (int bx=0; bx<Nt; bx++){
        rho=1.0;
        for (int by=0; by<Ns; by++){
            rho*=rho_vals[bx][by];
        }
        if (rand1[bx]<rho){
            for (int by=0; by<Ns; by++){
                k[bx][by][1]+=rand2[bx];
            }
        }
    }
}

__global__ void update(){
    
    int i=threadIdx.x, j=threadIdx.y, n=Nt*Ns;
    
    rand1 = new double[n];
    rand2 = new int[n];
    int p=Nt*threadIdx.y+threadIdx.x;
    
    curandState state;
    curand_init((unsigned long long)clock() + p, 0, 0, &state);
    
    rand1[p] = curand_uniform_double(&state);
    rand2[p] = 2*((int)(2*(1-curand_uniform_double(&state))))-1;
    
    if (i%2==0) a_update(0);
    __syncthreads();
    
    if (i%2==1) a_update(0);
    __syncthreads();
        
    rand1[p] = curand_uniform_double(&state);
    rand2[p] = 2*((int)(2*(1-curand_uniform_double(&state))))-1;
    
    if (j%2==0) a_update(1);
    __syncthreads();
    
    if (j%2==1) a_update(1);
    __syncthreads();
    
    
    rand1[p] = curand_uniform_double(&state);
    rand2[p] = 2*((int)(2*(1-curand_uniform_double(&state))))-1;
    
    if (i%2==0 && j%2==0) plaquette_update();
    __syncthreads();
    
    if (i%2==1 && j%2==0) plaquette_update();
    __syncthreads();
    
    if (i%2==0 && j%2==1) plaquette_update();
    __syncthreads();
    
    if (i%2==1 && j%2==1) plaquette_update();
    __syncthreads();
    
    rand1[p] = curand_uniform_double(&state);
    rand2[p] = 2*((int)(2*(1-curand_uniform_double(&state))))-1;
    
    temporalloop_update();
    __syncthreads();
    
    rand1[p] = curand_uniform_double(&state);
    rand2[p] = 2*((int)(2*(1-curand_uniform_double(&state))))-1;
    
    spatialloop_update();
    __syncthreads();
    
}

float phi2(int k[][Ns][d], int a[][Ns][d], double I[int_val]){
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
float phi4(int k[][Ns][d], int a[][Ns][d], double I[int_val]){
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
    
    double dmu=(mu_max-mu_min)/mu_n; 
    double n_avg, phi2_avg, phi4_avg;
    double xi[configs],phi2i[configs];
    mu=mu_min;
    
    for (int i=0; i<int_val; i++){
        I_val[i]=I(i);
    }
    
    ofstream data, data1;
    /*
    string filename="mu_n_phi2_phi4_Nt"+to_string(Nt)+"_Ns"+to_string(Ns)+"_eta"
        +to_string(eta)+"_lmd"+to_string(lmd)+"_"+to_string(configs)+"_"+to_string(gaps)+"_"
        +to_string(equil)+".txt";
    */
    data.open("mu_vs_n.txt");
    data1.open("mu_vs_phi2.txt");
    //data2.open("mu_vs_phi4.txt");
    
    dim3 threadblock = dim3(Nt, Ns);
    for (int g=0; g<mu_n; g++){
        
        for (int i=0; i<equil; i++){
            update<<<1,threadblock>>>();
            cudaDeviceSynchronize();
        }
        phi2_avg=0;
        n_avg=0;
        //phi4_avg=0;
        for (int i=0; i<configs; i++){
            for (int j=0; j<gaps; j++){
                update<<<1,threadblock>>>();
                cudaDeviceSynchronize();
                
            }
            update<<<1,threadblock>>>();
            cudaDeviceSynchronize();
            
            xi[i]=1.0*ksum(k)/Nt/Ns;
            phi2i[i]=phi2(k,a,I_val);
            
            n_avg+=xi[i];
            phi2_avg+=phi2i[i];
            //phi4_avg+=phi4(k,a,I_val);
        }
        n_avg=n_avg/configs;
        phi2_avg=phi2_avg/configs;
        //phi4_avg=phi4_avg/configs;
        
        data<<mu<<"\t"<<n_avg<<"\t"<<errorjack(xi);
        data1<<mu<<"\t"<<phi2_avg<<"\t"<<errorjack(phi2i);//<<"\t"<<phi4_avg<<endl;
        mu+=dmu;
        cout<<g<<endl;
        
    }
    
    data.close();
    data1.close();
    //data2.close();
    auto stop=high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-begin);
    
    cout<<duration.count()<<endl;
    
    return 0;
}
