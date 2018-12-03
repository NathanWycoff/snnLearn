#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <unistd.h>


// NOTE: Need to compile in C++11 mode, add -std=c++11
// These should eventually be specifiable from R
#define TAU 1
#define V_THRESH 1.5
#define THREADS_PER_BLOCK 512


// Integrated Postsynaptic Kernel
__host__ __device__
double ipostkern(double dt) {
    if (dt < 0) {
        return(0);
    }
    return(TAU * (1 - exp(-dt / TAU)));
}

// Postsynaptic Kernel
__host__ __device__
double postkern(double dt) {
    if (dt < 0) {
        return(0);
    }
    return(exp(-dt / TAU));
}

// Postsynaptic Kernel
__host__ __device__
double dpostkern(double dt) {
    if (dt < 0) {
        return(0);
    }
    return((-1.0) / TAU * exp(-dt / TAU));
}

// Integrated refractory kernel.
__host__ __device__
double iprekern(double dt) {
    if (dt < 0) {
        return(0);
    }
    return(-V_THRESH);
}

// The inner product function, uses the standard R^n inner product.
__host__ __device__
double inner_prod(double *x, double *y, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }   
    return(sum);
}

__global__
void par_c_main_loop(double ***ALPHA, double ***OMEGA, double ***Fcal, int **f_count, double ***Ws, int* net_shape, int n_layers, 
        int t_steps, double t_eps, int l, double ****GAMMA, double ****GAMMAd, const bool copy_gamma) {
    double t;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int n = index; n < net_shape[l]; n += stride) {
        t = 0;
        for (int ti = 0; ti < t_steps; ti++) {
            // Calculate total postsynaptic contribution 
            int n_f = f_count[l][n];
            double psc = 0;
            for (int tfi = 0; tfi < n_f; tfi++) {
                double tf = Fcal[l][n][tfi];
                psc += ipostkern(t - tf);
            }
            ALPHA[l][ti][n] = psc;

            if (l > 0) {
                // Update refractory contribution
                n_f = f_count[l][n];
                double ref = 0;
                for (int tfi = 0; tfi < n_f; tfi++) {
                    double tf = Fcal[l][n][tfi];
                    ref += iprekern(t - tf);
                }
                OMEGA[l-1][n][ti] = ref;

                // Update potential
                double V_n = inner_prod(Ws[l-1][n], ALPHA[l-1][ti], net_shape[l-1]) + OMEGA[l-1][n][ti];
                //printf("l = %d, n = %d, ti = %d", l, n, ti);
                //printf("Vsl = %d, n = %d, ti = %d", l, n, ti);

                // Check for firing neurons
                if (V_n > V_THRESH) {
                    // If an output fire, record the neural state
                    if (copy_gamma && l == n_layers-1) {
                        for (int l1 = 0; l1 < n_layers; l1++) {
                            for (int h = 0; h < net_shape[l1]; h++) {
                                GAMMA[n][f_count[l][n]][l1][h] = 0;
                                GAMMAd[n][f_count[l][n]][l1][h] = 0;
                                for (int ti = 0; ti < f_count[l1][h]; ti++) {
                                    double tf = Fcal[l1][h][ti];
                                    GAMMA[n][f_count[l][n]][l1][h] += postkern(t + t_eps - tf);
                                    GAMMAd[n][f_count[l][n]][l1][h] += dpostkern(t + t_eps - tf);
                                }
                            }
                        }
                    }
                    Fcal[l][n][f_count[l][n]] = t + t_eps;
                    f_count[l][n]++;
                }
            }
            t += t_eps;
        }
    }
}


// The main simulation, using armadillo for matrix multiplication, and organized in such a way that we solve a sequence embarassingly parallelizable problems.
double **par_sim_body_c(int *net_shape, const int n_layers,
        double **Fin, int *f_count_in, long long int **f_max, double ***Ws,
        int** f_count, const int t_steps, const double t_eps, double ****GAMMA, double ****GAMMAd, const int debug, const bool copy_gamma) {

    // Get the layer with the most neurons
    int max_neur = 0;
    for (int l = 0; l < n_layers; l++) {
        if (max_neur < net_shape[l]) {
            max_neur = net_shape[l];
        }
    }

    // ALPHA stores integrated postsynaptic potential in column major order.
    // OMEGA stores integrated refractory contribution in row major order.
    //double ***ALPHA = (double ***)calloc(n_layers, sizeof(double**));
    //double ***OMEGA = (double ***)calloc(n_layers-1, sizeof(double**));
    double ***ALPHA, ***OMEGA;
    cudaMallocManaged(&ALPHA, n_layers * sizeof(double**));
    cudaMallocManaged(&OMEGA, (n_layers-1) * sizeof(double**));
    for (int i = 0; i < n_layers; i++) {
        double **ALPHAi;
        cudaMallocManaged(&ALPHAi, t_steps * sizeof(double*));
        ALPHA[i] = ALPHAi;
        //ALPHA[i] = (double **) calloc(t_steps, sizeof(double*));
        for (int j = 0; j < t_steps; j++) {
            double *ALPHAij;
            cudaMallocManaged(&ALPHAij, net_shape[i] * sizeof(double));
            ALPHA[i][j] = ALPHAij;
            //ALPHA[i][j] = (double *) calloc(net_shape[i], sizeof(double));
        }
        if (i > 0) {
            double **OMEGAi;
            cudaMallocManaged(&OMEGAi, net_shape[i] * sizeof(double*));
            OMEGA[i-1] = OMEGAi;
            //OMEGA[i-1] = (double **) calloc(net_shape[i], sizeof(double*));
            for (int j = 0; j < net_shape[i]; j++) {
                double *OMEGAij;
                cudaMallocManaged(&OMEGAij, t_steps * sizeof(double));
                OMEGA[i-1][j] = OMEGAij;
                //OMEGA[i-1][j] = (double *) calloc(t_steps, sizeof(double));
            }
        }
    }

    if (debug >= 1) 
        printf("After ALPHA\n");

    // Storage for firing times
    //double ***u_Fcal = (double ***)calloc(n_layers, sizeof(double**));
    double ***u_Fcal;
    cudaMallocManaged(&u_Fcal, n_layers * sizeof(double**));

    // Copy input spike times to unified memory.
    double **u_Fin;
    cudaMallocManaged(&u_Fin, net_shape[0] * sizeof(double*));
    for (int n = 0; n < net_shape[0]; n++) {
        double *u_Finn;
        cudaMallocManaged(&u_Finn, f_count_in[n] * sizeof(double));
        cudaMemcpy(u_Finn, Fin[n], f_count_in[n] * sizeof(double), cudaMemcpyDefault);
        u_Fin[n] = u_Finn;
    }

    if (debug >= 1) 
        printf("After inputs \n");

    //int **myarr = (int **)malloc(2*sizeof(int *));
    //myarr[0] = (int **)malloc(2*sizeof(int));
    //myarr[1] = (int **)malloc(2*sizeof(int));
    //myarr[0][0] = 0;
    //myarr[0][1] = 1;
    //myarr[1][0] = 2;
    //myarr[1][1] = 3;

    //int **d_myarr;
    //cudaMallocManaged(&d_myarr, 2*sizeof(int *));
    //cudaMemcpy(d_myarr, myarr, 2*sizeof(int *), cudaMemcpyDefault);

    int **u_f_count;
    cudaMallocManaged(&u_f_count, n_layers * sizeof(int *));

    int *u_f_count_in;
    cudaMallocManaged(&u_f_count_in, net_shape[0] * sizeof(int));
    cudaMemcpy(u_f_count_in, f_count_in, net_shape[0] * sizeof(int), cudaMemcpyDefault);

    //f_count[0] = u_f_count_in;
    cudaMemcpy(&u_f_count[0], &u_f_count_in, sizeof(int *), cudaMemcpyDefault);
    u_Fcal[0] = u_Fin;
    for (int l = 0; l < n_layers-1; l++) {
        //double **Fi = (double **) calloc(net_shape[l+1], sizeof(double *));
        double **Fi;
        cudaMallocManaged(&Fi, net_shape[l+1] * sizeof(double *));
        u_Fcal[l+1] = Fi;

        //double **Fi = (double **) calloc(net_shape[l+1], sizeof(double *));
        int *f_countl;
        cudaMallocManaged(&f_countl, net_shape[l+1] * sizeof(int));
        cudaMemcpy(&u_f_count[l+1], &f_countl, sizeof(int *), cudaMemcpyDefault);
        for (int n = 0; n < net_shape[l+1]; n++) {
            double *Fln;
            cudaMallocManaged(&Fln, f_max[l+1][n] * sizeof(double));
            Fi[n] = Fln;
            // Initialize storeage to -1, so any negative firing time means 
            for (int f = 0; f < f_max[l+1][n]; f++) {
                Fi[n][f] = -1;
            }
        }
    }

    if (debug >= 1) 
        printf("After Fi copy\n");

    //// Convert Connection weights to a C array
    //// Ws[i] is the ith layer, Ws[i][j] is the jth row of layer i,
    //// Ws[i][j][k] is the j,k element of layer i (row major ordering).
    //double ***Ws_c = (double***)calloc(net_size-1, sizeof(double**));
    //for (int l = 0; l < net_size-1; l++) {
    //    Ws_c[l] = (double**)calloc(net_shape[l], sizeof(double*));
    //    for (int n = 0; n < net_shape[l]; n++) {
    //        Ws_c[l][n] = Ws_in + wlo[l] + net_shape[l+1] * n;
    //    }
    //}


    // Do GAMMA(d)
    // d_GAMMA[on][fi][l]][[h] Gives the instantaneous postsynaptic current of neuron h of layer l to firing time fi of output neuron on.
    double ****d_GAMMA, ****d_GAMMAd;
    if (copy_gamma) {
        cudaMallocManaged(&d_GAMMA, (n_layers-1) * sizeof(double***));
        cudaMallocManaged(&d_GAMMAd, (n_layers-1) * sizeof(double***));
        for (int on = 0; on < net_shape[n_layers-1]; on++) {
            cudaMallocManaged(&d_GAMMA[on], f_max[n_layers-1][on] * sizeof(double **));
            cudaMallocManaged(&d_GAMMAd[on], f_max[n_layers-1][on] * sizeof(double **));
            for (int fi = 0; fi < f_max[n_layers-1][on]; fi++) {
                cudaMallocManaged(&d_GAMMA[on][fi], n_layers * sizeof(double*));
                cudaMallocManaged(&d_GAMMAd[on][fi], n_layers * sizeof(double*));
                for (int l = 0; l < n_layers; l++) {
                    cudaMallocManaged(&d_GAMMA[on][fi][l], net_shape[l] * sizeof(double));
                    cudaMallocManaged(&d_GAMMAd[on][fi][l], net_shape[l] * sizeof(double));
                    for (int h = 0; h < net_shape[l]; h++) {
                        d_GAMMA[on][fi][l][h] = -1;
                        d_GAMMAd[on][fi][l][h] = -1;
                    }
                }
            }
        }
        if (debug >= 1) 
            printf("Initted GAMMA storage \n");
    }


    // Copy weights to unified memory
    double ***u_Ws;
    cudaMallocManaged(&u_Ws, (n_layers-1) * sizeof(double**));
    for (int l = 0; l < n_layers-1; l++) {
        double **u_Wsl;
        cudaMallocManaged(&u_Wsl, (net_shape[l+1]) * sizeof(double*));
        u_Ws[l] = u_Wsl;
        for (int n = 0; n < net_shape[l+1]; n++) {
            double *u_Wsln;
            cudaMallocManaged(&u_Wsln, net_shape[l] * sizeof(double));
            cudaMemcpy(u_Wsln, Ws[l][n], net_shape[l] * sizeof(double), cudaMemcpyDefault);
            u_Ws[l][n] = u_Wsln;
        }
    }

    if (debug >= 1) 
        printf("After Weights copy\n");

    // Copy network shape to unified memory
    int *u_net_shape;
    cudaMallocManaged(&u_net_shape, n_layers * sizeof(int));
    cudaMemcpy(u_net_shape, net_shape, n_layers * sizeof(int), cudaMemcpyDefault);

    // Run actual inference
    //TODO: Should just be + 1
    int n_blocks = max_neur / THREADS_PER_BLOCK + 1;

    // Main Loop
    for (int l = 0; l < n_layers; l++) {
        if (debug >= 1) 
            printf(" Solving Layer %d...\n", l);

        par_c_main_loop<<<n_blocks, THREADS_PER_BLOCK>>>(ALPHA, OMEGA, u_Fcal, u_f_count, u_Ws, u_net_shape, n_layers, 
                t_steps, t_eps, l, d_GAMMA, d_GAMMAd, copy_gamma);

        cudaDeviceSynchronize();
    }

    if (debug >= 1) 
        printf("After main loop\n");

    // Clean up
    for (int i = 0; i < n_layers; i++) {
        for (int j = 0; j < t_steps; j++) {
            cudaFree(ALPHA[i][j]);
        }
        cudaFree(ALPHA[i]);
        if (i > 0) {
            for (int j = 0; j < net_shape[i]; j++) {
                cudaFree(OMEGA[i-1][j]);
            }
            cudaFree(OMEGA[i-1]);
        }
    }
    cudaFree(ALPHA);
    cudaFree(OMEGA);

    if (debug >= 1) 
        printf("After Free\n");

    // Copy Fcal to host memory
    //double ***Fcal = (double ***)malloc(n_layers * sizeof(double **));
    //for (int l = 0; l < n_layers; l++) {
    //    Fcal[l] = (double **)malloc(net_shape[l] * sizeof(double *));
    //    for (int n = 0; n < net_shape[l]; n++) {
    //        Fcal[l][n] = (double *)malloc(f_max[l][n] * sizeof(double));
    //        cudaMemcpy(Fcal[l][n], u_Fcal[l][n], f_max[l][n] * sizeof(double), cudaMemcpyDefault);
    //    }
    //}
    // Copy output spikes to host memory

    double **Fout = (double **)malloc(net_shape[n_layers-1]*sizeof(double*));
    for (int n = 0; n < net_shape[n_layers-1]; n++) {
        Fout[n] = (double *)malloc(f_max[n_layers-1][n] * sizeof(double));
        cudaMemcpy(Fout[n], u_Fcal[n_layers-1][n], f_max[n_layers-1][n] * sizeof(double), cudaMemcpyDefault);
    }

    // Copy f_count to host memory
    for (int l = 0; l < n_layers; l++) {
        f_count[l] = (int *)malloc(net_shape[l] * sizeof(int));
        cudaMemcpy(f_count[l], u_f_count[l], net_shape[l] * sizeof(int), cudaMemcpyDefault);
    }

    if (debug >= 1) 
        printf("After output spike spike/f_count\n");

#if false
    // Copy to host memory
    // d_GAMMA[on][fi][l]][[h] Gives the instantaneous postsynaptic current of neuron h of layer l to firing time fi of output neuron on.
    //GAMMA = (double****)malloc((n_layers-1) * sizeof(double***));
    //GAMMAd = (double****)malloc((n_layers-1) * sizeof(double***));
    for (int on = 0; on < net_shape[n_layers-1]; on++) {
        GAMMA[on] = (double***)malloc(f_max[n_layers-1][on] * sizeof(double**));
        GAMMAd[on] = (double***)malloc(f_max[n_layers-1][on] * sizeof(double**));
        for (int fi = 0; fi < f_max[n_layers-1][on]; fi++) {
            GAMMA[on][fi] = (double**)malloc(n_layers * sizeof(double*));
            GAMMAd[on][fi] = (double**)malloc(n_layers * sizeof(double*));
            for (int l = 0; l < n_layers; l++) {
                GAMMA[on][fi][l] = (double*)malloc(net_shape[l] * sizeof(double));
                GAMMAd[on][fi][l] = (double*)malloc(net_shape[l] * sizeof(double));
                cudaMemcpy(GAMMA[on][fi][l], d_GAMMA[on][fi][l], net_shape[l] * sizeof(double), cudaMemcpyDefault);
                cudaMemcpy(GAMMAd[on][fi][l], d_GAMMAd[on][fi][l], net_shape[l] * sizeof(double), cudaMemcpyDefault);
            }
        }
    }

#endif

    if (debug >= 1) 
        printf("After GAMMA copy\n");

    //TODO: copy f_count

    return(Fout);
}
