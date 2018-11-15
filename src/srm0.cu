#include <iostream>
#include <algorithm>
#include <R.h>
#include <Rinternals.h>

using namespace std;

// These should eventually be specifiable from R
#define TAU 1
#define V_THRESH 1.5
#define t_eps 0.1

extern "C" void gvectorAdd(double *Ws_in, int *net_shape, int *net_sizep, double *Fin, int *f_count_in, int *f_max_R);

// Integrated Postsynaptic Kernel
__host__ __device__
double ipostkern(double dt) {
    if (dt < 0) {
        return(0);
    }
    return(TAU * (1 - exp(-dt / TAU)));
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
void par_c_main_loop(double ***Vs, double ***ALPHA, double ***OMEGA, double ***Fcal, int **f_count, double ***Ws, int* net_shape, int n_layers, 
        int t_steps) {
    double t;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int l = 0; l < n_layers; l++) {
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
                    Vs[l-1][n][ti+1] = inner_prod(Ws[l-1][n], ALPHA[l-1][ti], net_shape[l-1]) + OMEGA[l-1][n][ti];

                    // Check for firing neurons
                    if (Vs[l-1][n][ti+1] > V_THRESH) {
                        Fcal[l][n][f_count[l][n]] = t + t_eps;
                        f_count[l][n]++;
                    }
                }
                t += t_eps;
            }
        }
    }
}

// The main simulation, using armadillo for matrix multiplication, and organized in such a way that we solve a sequence embarassingly parallelizable problems.
double ***par_sim_body_c(int *net_shape, int n_layers,
        double **Fin, int *f_count_in, long long int **f_max, double ***Ws,
        int** f_count) {
    // Do simulation
    int t_steps = 35;

    printf("333");

    // Print some info about params passed
    printf("n_layers: %d\n", n_layers);
    for (int l = 0; l < n_layers; l++) {
        printf("net_shape: %d\n", net_shape[l]);
    }
    printf("t_steps: %d\n", t_steps);

    // Stores electric potential for each layer in row major order.
    //double ***Vs = (double ***)calloc(n_layers-1, sizeof(double**));
    double ***Vs;
    cudaMallocManaged(&Vs, (n_layers-1) * sizeof(double **));
    printf(" %i ", n_layers-1);
    printf("|| %p  ||", Vs[0]);
    printf("a");
    for (int i = 0; i < n_layers-1; i++) {
        printf("b");
        double **Vsi;
        cudaMallocManaged(&Vsi, net_shape[i+1] * sizeof(double*));
        Vs[i] = Vsi;
        //Vs[i] = (double **)calloc(net_shape[i+1], sizeof(double*));
        for (int j = 0; j < net_shape[i+1]; j++) {
            printf("c");
            double *Vsij;
            cudaMallocManaged(&Vsij, (t_steps+1) * sizeof(double*));
            printf("c1");
            Vsij[0] = 0;// Initialize voltage at resting potential, assumed 0.
            printf("d");
            Vs[i][j] = Vsij;
            printf("e");
            //Vs[i][j] = (double *)calloc(t_steps + 1, sizeof(double));
        }
    }

    printf("AAAA");

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

    printf("BBBB");

    // Storage for firing times
    //double ***Fcal = (double ***)calloc(n_layers, sizeof(double**));
    double ***Fcal;
    cudaMallocManaged(&Fcal, n_layers * sizeof(double**));

    // Copy input spike times to unified memory.
    double **u_Fin;
    cudaMallocManaged(&u_Fin, net_shape[0] * sizeof(double*));
    for (int n = 0; n < net_shape[0]; n++) {
        double *u_Finn;
        cudaMallocManaged(&u_Finn, f_count_in[n] * sizeof(double));
        cudaMemcpy(u_Finn, Fin[n], net_shape[0] * sizeof(double), cudaMemcpyDefault);
        u_Fin[n] = u_Finn;
    }

    printf("CCCC");

    int *u_f_count_in;
    cudaMallocManaged(&u_f_count_in, net_shape[0] * sizeof(int));
    cudaMemcpy(u_f_count_in, f_count_in, net_shape[0] * sizeof(int), cudaMemcpyDefault);

    printf("444");

    printf("aaa");

    f_count[0] = u_f_count_in;
    Fcal[0] = u_Fin;
    for (int l = 0; l < n_layers-1; l++) {
        //double **Fi = (double **) calloc(net_shape[l+1], sizeof(double *));
        double **Fi;
        cudaMallocManaged(&Fi, net_shape[l+1] * sizeof(double *));
        Fcal[l+1] = Fi;

        int *f_countl;
        cudaMallocManaged(&f_countl, net_shape[l+1] * sizeof(int));
        f_count[l+1] = f_countl;
        //f_count[l+1] = (int *)calloc(net_shape[l+1], sizeof(int));
        for (int n = 0; n < net_shape[l+1]; n++) {
            double *Fin;
            cudaMallocManaged(&Fin, f_max[l+1][n] * sizeof(double));
            Fi[n] = Fin;
            // Initialize storeage to -1, so any negative firing time means 
            //Fi[n] = (double *) calloc(f_max[l+1][n], sizeof(double));
        }
    }

    printf("bbb");

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


    // Copy weights to unified memory
    double ***u_Ws;
    cudaMallocManaged(&u_Ws, (n_layers-1) * sizeof(double**));
    for (int l = 0; l < n_layers-1; l++) {
        double **u_Wsl;
        cudaMallocManaged(&u_Wsl, (net_shape[l]) * sizeof(double*));
        u_Ws[l] = u_Wsl;
        for (int n = 0; n < net_shape[l]; n++) {
            double *u_Wsln;
            cudaMallocManaged(&u_Wsln, net_shape[l] * sizeof(double));
            cudaMemcpy(u_Wsln, Ws[l][n], net_shape[l] * sizeof(double), cudaMemcpyDefault);
            u_Ws[l][n] = u_Wsln;
        }
    }

    printf("ccc");

    // Copy network shape to unified memory
    int *u_net_shape;
    cudaMallocManaged(&u_net_shape, n_layers * sizeof(int));
    cudaMemcpy(u_net_shape, net_shape, n_layers * sizeof(int), cudaMemcpyDefault);

    // Run actual inference
    par_c_main_loop<<<1, 1>>>(Vs, ALPHA, OMEGA, Fcal, f_count, u_Ws, u_net_shape, n_layers, 
            t_steps);

    printf("ddd");

    // Clean up
    for (int i = 0; i < n_layers-1; i++) {
        for (int j = 0; j < net_shape[i+1]; j++) {
            cudaFree(Vs[i][j]); 
        }
        cudaFree(Vs[i]);
    }
    cudaFree(Vs);

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

    printf("555");

    return(Fcal);
}


void gvectorAdd(double *Ws_in, int *net_shape, int *net_sizep, double *Fin, int *f_count_in, int *f_max_R) {
    // R only passes pointers.
    int net_size = *net_sizep;
    
    // Calculate the cumulative size of each layer's weight matrix
    int *wlo = (int *)calloc(net_size-1, sizeof(int *));
    wlo[0] = 0;
    printf("Offsets:\n");
    printf("%i\n", wlo[0]);
    for (int l = 0; l < net_size-2; l++) {
        wlo[l+1] = wlo[l] + net_shape[l] * net_shape[l+1];
        printf("%i\n", wlo[l+1]);
    }

    // Convert Connection weights to a C array
    // Ws[i] is the ith layer, Ws[i][j] is the jth row of layer i,
    // Ws[i][j][k] is the j,k element of layer i (row major ordering).
    double ***Ws_c = (double***)calloc(net_size-1, sizeof(double**));
    for (int l = 0; l < net_size-1; l++) {
        Ws_c[l] = (double**)calloc(net_shape[l], sizeof(double*));
        for (int n = 0; n < net_shape[l]; n++) {
            Ws_c[l][n] = Ws_in + wlo[l] + net_shape[l+1] * n;
        }
    }

    double **Fin_c = (double**)calloc(net_shape[0], sizeof(double *));
    int cum_in = 0;
    for (int n = 0; n < net_shape[0]; n++) {
        Fin_c[n] = &Fin[cum_in];
        cum_in += f_count_in[n];
    }

    printf("Firing Times:\n");
    for (int n = 0; n < net_shape[0]; n++) {
        printf("Neuron %i\n", n+1);
        for (int tf = 0; tf < f_count_in[n]; tf++) {
            printf("%f\n", Fin_c[n][tf]);
        }
    }


    int ** f_max = (int **)calloc(net_size, sizeof(int*));
    int cum_size = 0;
    for (int l = 0; l < net_size; l++) {
        f_max[l] = f_max_R + cum_size;
        cum_size += net_shape[l];
    }

    //// Calculate an upper bound on how many times each neuron will fire in C
    //int ** f_max = (int **)calloc(net_size, sizeof(int*));
    //f_max[0] = f_count_in;
    //for (int l = 0; l < net_size-1; l++) {
    //    //f_max[l+1] = conv_to<ivec>::from(TAU / V_THRESH * Ws[l].t() * f_max[l]);
    //    f_max[l+1] = (int *)calloc(net_shape[l+1], sizeof(int));
    //    for (int n1 = 0; n1 < net_shape[l+1]; n1++) {
    //        double accum = 0;
    //        for (int n2 = 0; n2 < net_shape[l]; n2++) {
    //            accum += TAU / V_THRESH * Ws_c[l][n2][n1] * f_max[l][n2];
    //        }
    //        f_max[l+1][n1] = (int) accum;
    //    }
    //}

    // Convert to a C array for use with GPU
    long long int **f_max_c = 
        (long long int **)calloc(net_size, sizeof(long long int*));
    for (int l = 0; l < net_size; l++) {
        f_max_c[l] = (long long int *)calloc(net_shape[l], sizeof(long long int));
        for (int n = 0; n < net_shape[l]; n++) {
            f_max_c[l][n] = (long long int) f_max[l][n];
        }
    }

    //printf("Max Fire Counts:\n");
    //for (int l = 0; l < net_size; l++) {
    //    printf("Layer %i\n", l);
    //    for (int n = 0; n < net_shape[l]; n++) {
    //        printf("%i\n", f_max_c[l][n]);
    //    } 
    //}

    // Do SRM0 simulation
    double ***Fcal;
    //int **f_count = (int **)calloc(net_shape.size(), sizeof(int *));
    int **f_count;
    cudaMallocManaged(&f_count, net_size * sizeof(int *));
    printf("111");
    Fcal = par_sim_body_c(net_shape, net_size, Fin_c, 
            f_count_in, f_max_c, Ws_c, f_count);
    printf("222");

    // Print out the results
    for (int l = 0; l < net_size; l++) {
        cout << "Layer:" << l << endl;
        for (int n = 0; n < net_shape[l]; n++) {
            for (int f = 0; f < f_count[l][n]; f++) {
                cout << "Value:" << endl;
                cout << Fcal[l][n][f] << endl;
            }
        }
    }
    

    for (int l = 0; l < net_size; l++) {
        for (int n = 0; n < net_shape[l]; n++) {
            cout << f_max[l][n] << endl;
        }
    }

    //TODO: free things at some point.

}
