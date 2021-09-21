
/* REFER TO THE PAPER arXiv:1711.02311v1 [hep-lat] 7 Nov 2017 FOR NOTATIONS */
 
#define Ns 10
#define Nt 10
#define d 2    

/*
    -'Ns' and 'Nt' are the spatial and temporal dimensions of the lattice.
    -'d' is the number of dimentions of the lattice.
*/

double eta=4.01, lmd=1.0;
const int configs=4000,gaps=10,equil=2000;
double dr=0.01, inf=10, mu_min=0.0, mu_max=1.3;  
const int mu_n=100, int_val=3000;

/*
    -'eta' and 'lmd' are coefficients of |phi_x|^2 and |phi_x|^4 from eq(1) of the paper.
    -'configs' is the number of conigurations used to calculate the observables,
    -'gaps' is the iterations discarded between the measured configurations for decorrelation,
    -'equil' is the number of iterations for thermalization.
    -'int_val', 'inf' and 'dr' are parameters to calculate I(s_x) from eq(3) of the paper.
    -'mu_min' and 'mu_max' are the minimum and maximum values of the range of chemical potentials.
    -'mu_n' is the number of chemical potentials values in the above range.
*/
