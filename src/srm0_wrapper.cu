/** src/srm0_wrapper.c Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.21.2018  */

// An R interface to srm0.cu
#include <R.h>
#include <Rinternals.h>
#include <unistd.h>
#include "srm0.h"

extern "C" void gvectorAdd(double *Ws_in, int *net_shape, int *net_sizep, double *Fin, int *f_count_in, int *f_max_R, double *Flast);
//extern void gvectorAdd(double *Ws_in, int *net_shape, int *net_sizep, double *Fin, int *f_count_in, int *f_max_R);

void gvectorAdd(double *Ws_in, int *net_shape, int *net_sizep, double *Fin, int *f_count_in, int *f_max_R, double *Flast) {
    // R only passes pointers.
    int net_size = *net_sizep;
    
    // Calculate the cumulative size of each layer's weight matrix
    int *wlo = (int *)calloc(net_size-1, sizeof(int *));
    wlo[0] = 0;
    for (int l = 0; l < net_size-2; l++) {
        wlo[l+1] = wlo[l] + net_shape[l] * net_shape[l+1];
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

    // Do SRM0 simulation
    double ***Fcal;
    //int **f_count = (int **)calloc(net_shape.size(), sizeof(int *));
    int **f_count;
    cudaMallocManaged(&f_count, net_size * sizeof(int *));
    Fcal = par_sim_body_c(net_shape, net_size, Fin_c, 
            f_count_in, f_max_c, Ws_c, f_count);

    // Print out the results
    //for (int l = 0; l < net_size; l++) {
    //    cout << "Layer:" << l << endl;
    //    for (int n = 0; n < net_shape[l]; n++) {
    //        for (int f = 0; f < f_count[l][n]; f++) {
    //            cout << "Value:" << endl;
    //            cout << Fcal[l][n][f] << endl;
    //        }
    //    }
    //}
    

    //for (int l = 0; l < net_size; l++) {
    //    for (int n = 0; n < net_shape[l]; n++) {
    //        cout << f_max[l][n] << endl;
    //    }
    //}

    // Pack up the last layer's firing times.
    int *cum_fires = (int *)calloc(net_shape[net_size-1], sizeof(int));
    for (int n = 0; n < net_shape[net_size-1]; n++) {
        if (n == 0) {
            cum_fires[n] = 0;
        } else {
            cum_fires[n] = cum_fires[n-1] + f_max[net_size-1][n-1];
        }
    }

    //Flast = (double *)calloc(cum_fires[net_shape[net_size-1]], sizeof(double));
    for (int n = 0; n < net_shape[net_size-1]; n++) {
        for (int f = 0; f < f_max[net_size-1][n]; f++) {
            Flast[f + cum_fires[n]] = Fcal[net_size-1][n][f];
        }
    }

    //TODO: free things at some point.
}
