//Sebastián Luciano Gallardo -- Octubre 2020
//This library defines the floquet class, which lets define and solve a time dependent quantum mechanics problem with Hamiltonian of the form H(t)=H0+V*A*f(w*t) in the Floquet formalism, where V is an arbitrary hermitian matrix, A and w are real constants, and f is TAU periodic.
//H0 is allowed to depend in a set of parameters which can be set via a setter function.
//A and w can also be modified at will through setter functions.
//The problem is described in a basis where V is diagonal w/o loss of generality.
//Implements functions for obtaining time dependent and time averaged probability transitions between states, quasienergies, Floquet eigenstates, concurrence.

#define NK 128
 //discretization of driving period
#define TAU 6.283185307179586 //=2pi

//Libraries
#include <cstdlib>
#include <math.h>
#include <complex>
#include <armadillo>
#include <fftw3.h>

#define C1 (cx_double){1,0}
#define CI (cx_double){0,1}
#define C0 (cx_double){0,0}

using namespace std;
using namespace arma;

//Pauli Matrices
#ifndef PAULI_SIGMA_MATRICES
#define PAULI_SIGMA_MATRICES
const cx_mat ID = {{C1,C0},{C0,C1}};
const cx_mat SX = {{C0,C1},{C1,C0}};
const cx_mat SY = {{C0,-CI},{CI,C0}};
const cx_mat SZ = {{C1,C0},{C0,-C1}};
const cx_mat SP = {{C0,C0},{C1,C0}};
const cx_mat SM = {{C0,C1},{C0,C0}};
#endif

class floquet{
    public:
    int dim, npar; //dim: Hamiltonian dimension, npar: number of H0 parameters
    cx_mat (*H0_func)(double*); //gives H0 from par
    vec V; //driving operator
    double (*f)(double);  //driving function (TAU-periodic)

    //Hamiltonian parameters
    double *par;   //vector of H0 + A + w parameters
    double A, w;

    //Solution Output (H0)
    cx_mat H0;
    cx_mat eigst; //of H0, in ascending energy order
    vec energy;

    //Solution Output (Floquet)
    cx_mat *flst_k;  //flEigenstate[NK][dim,dim]
    cx_mat *flst_t;
    vec qenergy;

    //functions for calculating solutions
    void calcH0();
    void calcFloquet();

    //functions for yielding output
    double transProbAvg(cx_vec psi_i, cx_vec psi_f);
    double transProb(cx_vec psi_i, cx_vec psi_f, double t0, double t);
    cx_vec propagate(cx_vec psi0, double t0, double t);
    cx_vec propagate_discrete(cx_vec psi0, int t0, int t); //units measured in time discretization
    cx_mat mixingMatrix(int par_i, double par_epsilon);

    //constructors/destructors
    floquet(cx_mat(*H0_funcI)(double*), vec VI, double(*fI)(double), int dimI, int nparI):
        H0_func(H0_funcI),
        V(VI),
        f(fI),
        dim(dimI),
        npar(nparI)
        {
        par = new double[nparI];
        flst_k = new cx_mat[NK];
        flst_t = new cx_mat[NK];
        }
    ~floquet()
        {
        delete[] par;
        delete[] flst_k;
        delete[] flst_t;
        }

};

//Class functions


void floquet::calcH0(){   //calculates and diagonalizes H0
    H0 = H0_func(par);
    eigst = zeros<cx_mat>(dim,dim);
    energy = zeros<vec>(dim);
    eig_sym(energy,eigst,symmatu(H0));//symmatu is to avoid non-hermiticity caused by rounding errors.
}

void floquet::calcFloquet(){   //here is the only place where we use that V is diagonal
    //Trotter Suzuki 4th order

    for(int i=0;i<NK;i++){
        flst_k[i] = zeros<cx_mat>(dim,dim);
        flst_t[i] = zeros<cx_mat>(dim,dim);
    }
    qenergy = zeros<vec>(dim);


    #define s (1./(4.-pow(4.,1./3.)))
    #define z (1-4*s)

    int i,j,k,it;
    double aux, t;
    double dt=TAU/w/NK;
    // double psi[NK][2];
    cx_vec psi(NK);
    cx_vec psi_2(NK);

    cx_cube u(dim,dim,NK+1,fill::zeros);
    cx_mat ctu(dim,dim,fill::zeros);
    for(i=0; i<dim; i++) u(i,i,0)=C1;


    for(it=0; it<NK; it++){
        double v1=A*f(w*(it*dt+s*dt/2));
        double v2=A*f(w*(it*dt+3*s*dt/2));
        double v3=A*f(w*(it*dt+2*s*dt+z*dt/2));
        double v4=A*f(w*(it*dt+2*s*dt+z*dt+s*dt/2));
        double v5=A*f(w*((it+1)*dt-s*dt/2));
        cx_mat uv1_(dim,dim,fill::zeros);
        cx_mat uv2_(dim,dim,fill::zeros);
        cx_mat uv3_(dim,dim,fill::zeros);
        cx_mat uv4_(dim,dim,fill::zeros);
        cx_mat uv5_(dim,dim,fill::zeros);
        cx_mat uv6_(dim,dim,fill::zeros);
        for(i=0;i<dim;i++){
            uv1_(i,i)=exp(-CI*V[i]*(v1*dt*s/2));
            uv2_(i,i)=exp(-CI*V[i]*(v1*dt*s/2+v2*dt*s/2));
            uv3_(i,i)=exp(-CI*V[i]*(v2*dt*s/2+v3*dt*z/2));
            uv4_(i,i)=exp(-CI*V[i]*(v3*dt*z/2+v4*dt*s/2));
            uv5_(i,i)=exp(-CI*V[i]*(v4*dt*s/2+v5*dt*s/2));
            uv6_(i,i)=exp(-CI*V[i]*(v5*dt*s/2));
        }

        cx_mat uh1_(dim,dim,fill::zeros);
        cx_mat uh2_(dim,dim,fill::zeros);
        for(i=0;i<dim;i++){
            cx_mat tensorprod = kron(eigst.col(i).t(),eigst.col(i));
            uh1_+=exp(-CI*energy[i]*dt*s)*tensorprod;
            uh2_+=exp(-CI*energy[i]*dt*z)*tensorprod;
        }

        u.slice(it+1)=uv6_*uh1_*uv5_*uh1_*uv4_*uh2_*uv3_*uh1_*uv2_*uh1_*uv1_*u.slice(it);
    }

    cx_mat id(dim,dim,fill::eye);
    ctu = symmatu(CI*inv(id-u.slice(NK))*(id+u.slice(NK)));
    eig_sym(qenergy,ctu,ctu);
    for(i=0;i<dim;i++) qenergy[i]=2*atan(1/qenergy[i])/NK/dt;

    //we have to sort again the quasienergies and eigenstates (slightly lazy implementation)
    cx_mat ctu_sorted(dim,dim,fill::none);
    vec qenergy_sorted(dim,fill::none);
    uvec sorted_idx(dim,fill::none);
    sorted_idx = sort_index(qenergy);
    for(i=0;i<dim;i++){
        qenergy_sorted(i)=qenergy(sorted_idx(i));
        ctu_sorted.col(i)=ctu.col(sorted_idx(i));
    }
    qenergy=qenergy_sorted;
    ctu=ctu_sorted;


    t=0;
    for(j=0;j<dim;j++)
        for(i=0;i<dim;i++)
            flst_t[0](i,j)=ctu(i,j);
    for(it=1;it<=NK;it++){
        t=t+dt;
        for(j=0;j<dim;j++){
            for(i=0;i<dim;i++){
                flst_t[it%NK](i,j)=C0;
                for(k=0;k<dim;k++){
                    aux=qenergy[j]*t;
                    flst_t[it%NK](i,j)+=u(i,k,(it-1)%NK)*ctu(k,j)*((cx_double){cos(aux),sin(aux)});
                }
            }
        }
    }

    //obtained floquet states at different times, now let's calculate their fourier counterparts

    fftw_plan plan = fftw_plan_dft_1d(NK,(double(*)[2])&psi(0),(double(*)[2])&psi(0), FFTW_FORWARD, FFTW_MEASURE);
    for(j=0;j<dim;j++)
        for(i=0;i<dim;i++){
            for(k=0;k<NK;k++) psi(k) = flst_t[k](i,j);
            fftw_execute(plan);
            for(k=0;k<NK;k++) flst_k[k](i,j)= psi(k)/((double)NK);
        }
    fftw_destroy_plan(plan);

}


double floquet::transProbAvg(cx_vec psi_i, cx_vec psi_f){ //t and t0 - averaged transition probability from |psi_i> -> |psi_f>
    double result=0;
    int i,k1,k2;
    for(i=0;i<dim;i++)
        for(k1=0;k1<NK;k1++)
            for(k2=0;k2<NK;k2++)
                result+=std::norm(cdot(psi_i,flst_k[k1].col(i)))*std::norm(cdot(psi_f,flst_k[k2].col(i)));
    return result;
}


cx_vec floquet::propagate(cx_vec psi_i, double t0, double t){
    //Calculates the wavefunction that is psi0 at time 0 at time t.
    //using linear interpolation of the flst_t
    cx_vec psi_f(dim,fill::zeros);
    double timestep = TAU/NK/w;
    int it = ((int)(t/timestep));
    int it0 = ((int)(t0/timestep));
    double d = t/timestep-it;
    double d0 = t0/timestep-it0;
    cx_vec flt(dim,fill::none);
    cx_vec flt0(dim,fill::none);
    for(int a=0;a<dim;a++){
        flt = (1-d)*flst_t[it%NK].col(a)+d*flst_t[(it+1)%NK].col(a);
        flt0 = (1-d0)*flst_t[it0%NK].col(a)+d0*flst_t[(it0+1)%NK].col(a);
        psi_f+=exp(-CI*qenergy(a)*(t-t0))*cdot(flt0,psi_i)*flt;
    }
    return normalise(psi_f);
}

cx_vec floquet::propagate_discrete(cx_vec psi_i, int t0, int t){ //in units of time discretization, way more efficient
    cx_vec psi_f(dim,fill::zeros);
    double dt = TAU/NK/w;
    for(int a=0;a<dim;a++)
        psi_f += exp(-CI*qenergy(a)*((t-t0)*dt))*cdot(flst_t[t0%NK].col(a),psi_i)*flst_t[t%NK].col(a);
    return normalise(psi_f);
}


cx_mat floquet::mixingMatrix(int par_i, double par_epsilon){ //calculates the hermitian matrix of eigenstate mixing
    double par_neighbor[npar];
    for(int i=0;i<npar;i++) par_neighbor[i] = par[i];
    par_neighbor[par_i] += par_epsilon;
    cx_mat eigenstate_neighbor(dim,dim,fill::none);
    vec eigenenergy_neighbor(dim,fill::none);
    cx_mat H0_neighbor = H0_func(par_neighbor);
    eig_sym(eigenenergy_neighbor,eigenstate_neighbor,symmatu(H0_neighbor)); //symmatu is to avoid non-hermiticity caused by rounding errors.
    return -CI*(eigenstate_neighbor*eigst.t()-eye<cx_mat>(dim,dim))/par_epsilon;
}







//not working right! use propagate instead

// double floquet::transProb(cx_vec psi_i, cx_vec psi_f, double t0, double t){ //time dependent transition probability from |psi_i> -> |psi_f>
//     //using linear interpolation of flst_ts
//     //time measured in units of driving period
//     cx_double cumsum = C0;
//     double timestep = TAU/NK/w;
//     int it = ((int)(t/timestep));
//     int it0 = ((int)(t0/timestep));
//     double d = t/timestep-it;
//     double d0 = t0/timestep-it0;
//     cx_vec flt(dim,fill::none);
//     cx_vec flt0(dim,fill::none);
//     for(int a=0;a<dim;a++){
//         flt = (1-d)*flst_t[it%NK].col(a)+d*flst_t[(it+1)%NK].col(a);
//         flt0 = (1-d0)*flst_t[it0%NK].col(a)+d0*flst_t[(it0+1)%NK].col(a);
//         cumsum+=exp(-CI*qenergy[a]*(t-t0))*cdot(psi_f,flt)*cdot(flt0,psi_i);
//     }
//     return norm(cumsum);
// }





// int floquet::setParameters(double *parI){   //tries to update parameters performing minimal calculations.
//     for(int i=0;i<npar;i++)
//         if(par[i]!=parI[i]){
//             for(int j=0;j<npar;j++) par[j]=parI[j];
//             calcH0();
//             calcFloquet();
//             return 1; //if modifications were made
//         }
//     return 0; //if no modifications were made
// }


// double floquet::concurrence4d_avg_pure(double *parI, cx_vec psi0){ //for pure states
//     //parameters par + A + W
//     setParameters(parI);
//     double c1,c2;
//     cx_double aux;
//     double dt = TAU/w/NK;
//     double w = w;
//     long nt_step, it;
//     //set step in time integration
//     //the largest timescale is given by the smallest difference between quasienergies
//     double mindq = w+qenergy[0]-qenergy[3];
//     for(int i=1;i<4;i++) if((c1=(qenergy[i]-qenergy[i-1]))<mindq) mindq=c1;
//     nt_step = (long)(TAU/mindq)*NK;
//     c1=0;
//     cx_vec psi_t(dim,fill::none);

//     //do the thing
//     for(long it0=0;it0<NK;it0++){//average over initial condition time t0
//         c2=0;
//         it=it0;
//         for(long i=0;i<NT;i++){
//             it+=nt_step+rand()%nt_step;
//             aux=C0;
//             for(int a=0;a<dim;a++)
//                 for(int b=0;b<dim;b++)
//                     aux+=cdot(flst_t[it0%NK].col(a),psi0)*cdot(flst_t[it0%NK].col(b),psi0)*exp(-CI*(qenergy[a]+qenergy[b])*((it-it0)*dt))*cdot(conj(flst_t[it%NK].col(a)),kron(SY,SY)*flst_t[it%NK].col(b));
//             c2+=sqrt(norm(aux));
//         }
//         c1+=c2/NT;
//     }
//     c1=c1/NK;
//     return c1;
// }


// double floquet::concurrence4d_avg_rho(double *parI, cx_vec psi0){ //with density matrix
//     //parameters par + A + W
//     setParameters(parI);
//     double c1,c2;
//     double dt = TAU/w/NK;
//     double w = w;
//     long nt_step, it;
//     //set step in time integration
//     //the largest timescale is given by the smallest difference between quasienergies
//     double mindq = w+qenergy[0]-qenergy[3];
//     for(int i=1;i<4;i++) if((c1=(qenergy[i]-qenergy[i-1]))<mindq) mindq=c1;

//     nt_step = (long)(TAU/mindq)*NK;
//     c1=0;
//     cx_vec psi_t(dim,fill::none);

//     //do the thing
//     for(long it0=0;it0<NK;it0++){ //average over initial condition time t0
//         c2=0;
//         it=it0;
//         for(long i=0;i<NT;i++){
//             it+=nt_step+rand()%nt_step;
//             psi_t = zeros<cx_vec>(dim);
//             for(int a=0;a<dim;a++)
//                 psi_t += exp(-CI*qenergy[a]*((it-it0)*dt))*flst_t[it%NK].col(a)*cdot(flst_t[it0%NK].col(a),psi0);
//             c2+=concurrence4d(psi_t*psi_t.t());
//         }
//         c1+=c2/NT;
//     }
//     c1=c1/NK;
//     return c1;
// }




// double floquet::quasienergy(double *parI, int i){
//     //parameters: par + A + w
//     setParameters(parI);
//     return qenergy[i];
// }
// double floquet::concurrence4d_t(double *parI, cx_vec psi0, double t0){ //Calculates time-dependent concurrence for the state that is psi0 at t=t0.
//     //parameters par + A + W + t
//     setParameters(parI);
//     cx_vec psi_t = aux_wavefunc_t(psi0,parI[npar],t0);
//     return concurrence4d_pure(psi_t);
// }

// double floquet::concurrence4d_t_t0(double *parI, cx_vec psi0){ //Calculates time-dependent concurrence for the state that is psi0 at t=t0.
//     //parameters par + A + W + t + t0
//     setParameters(parI);
//     cx_vec psi_t = aux_wavefunc_t(psi0,parI[npar],parI[npar+1]);
//     return concurrence4d_pure(psi_t);
// }
