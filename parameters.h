
/* REFER TO THE PAPER arXiv:1711.02311v1 [hep-lat] 7 Nov 2017 FOR NOTATIONS */
 
#define Ns 10
#define Nt 10
#define d 2    

/*
    -'Ns' and 'Nt' are the spatial and temporal dimensions of the lattice.
    -'d' is the number of dimentions of the lattice.
*/

double eta=4.01, lmd=1.0;
const int configs=2,gaps=0,equil=0;
double dr=0.01, inf=10, mu_min=0.0, mu_max=1.0;  
const int mu_n=20, int_val=3000;

/*
    -'eta' and 'lmd' are coefficients of |phi_x|^2 and |phi_x|^4 from eq(1) of the paper.
    -'configs' is the number of conigurations used to calculate the observables,
    -'gaps' is the iterations discarded between the measured configurations for decorrelation,
    -'equil' is the number of iterations for thermalization.
    -'int_val', 'inf' and 'dr' are parameters to calculate I(s_x) from eq(3) of the paper.
    -'mu_min' and 'mu_max' are the minimum and maximum values of the range of chemical potentials.
    -'mu_n' is the number of chemical potentials values in the above range.
*/


__managed__ int k[Nt][Ns][d]={0}, a[Nt][Ns][d]={0}, a_[Nt][Ns][d]={0};
__managed__ double I_val[int_val], mu;
__shared__ double *rand1;
__shared__ int *rand2;

/*
    -'k' and 'a' arrays store the k_x,v and a_x,v variables from the paper.
    -'I_val' stores I(s_x) from eq(3) of the paper for different values of s_x.
    -other variables are just for the sake of simulation.
*/