/** src/srm0_wrapper.c Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.21.2018  */

// An R interface to srm0.cu
#include <R.h>
#include <Rinternals.h>
#include <unistd.h>
#include "srm0.h"

extern "C" void gvectorAdd(double *Ws_in, int *net_shape, int *n_layersp, double *Fin, int *f_count_in, int *f_max_R, double *Flast, int *t_steps_R, double *t_eps_R);
//extern void gvectorAdd(double *Ws_in, int *net_shape, int *n_layersp, double *Fin, int *f_count_in, int *f_max_R);

void gvectorAdd(double *Ws_in, int *net_shape, int *n_layersp, double *Fin, int *f_count_in, int *f_max_R, double *Flast, int *t_steps_R, double *t_eps_R) {
    // R only passes pointers.
    int n_layers = *n_layersp;
    int t_steps = *t_steps_R;
    double t_eps = *t_eps_R;
    
    // Calculate the cumulative size of each layer's weight matrix
    int *wlo = (int *)calloc(n_layers-1, sizeof(int *));
    wlo[0] = 0;
    for (int l = 0; l < n_layers-2; l++) {
        wlo[l+1] = wlo[l] + net_shape[l] * net_shape[l+1];
    }

    // Convert Connection weights to a C array
    // Ws[i] is the ith layer, Ws[i][j] is the jth col of layer i,
    // Ws[i][j][k] is the k,j element of layer i (col major ordering).
    // Ws[i][j] gives all input weights to neuron j of layer i
    double ***Ws_c = (double***)calloc(n_layers-1, sizeof(double**));
    for (int l = 0; l < n_layers-1; l++) {
        Ws_c[l] = (double**)calloc(net_shape[l+1], sizeof(double*));
        for (int n = 0; n < net_shape[l+1]; n++) {
            Ws_c[l][n] = Ws_in + wlo[l] + net_shape[l] * n;
        }
    }

    // Count number of input firings for each input neuron
    double **Fin_c = (double**)calloc(net_shape[0], sizeof(double *));
    int cum_in = 0;
    for (int n = 0; n < net_shape[0]; n++) {
        Fin_c[n] = &Fin[cum_in];
        cum_in += f_count_in[n];
    }

    int ** f_max = (int **)calloc(n_layers, sizeof(int*));
    int cum_size = 0;
    for (int l = 0; l < n_layers; l++) {
        f_max[l] = f_max_R + cum_size;
        cum_size += net_shape[l];
    }

    //// Calculate an upper bound on how many times each neuron will fire in C
    //int ** f_max = (int **)calloc(n_layers, sizeof(int*));
    //f_max[0] = f_count_in;
    //for (int l = 0; l < n_layers-1; l++) {
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
        (long long int **)calloc(n_layers, sizeof(long long int*));
    for (int l = 0; l < n_layers; l++) {
        f_max_c[l] = (long long int *)calloc(net_shape[l], sizeof(long long int));
        for (int n = 0; n < net_shape[l]; n++) {
            f_max_c[l][n] = (long long int) f_max[l][n];
        }
    }

    printf("Input validation:\n");
    printf("n_layers: %d", n_layers);
    printf("net_shape:\n");
    for (int l = 0; l < n_layers; l++) {
        printf("l %d = %d\n", l, net_shape[l]);
    }
    printf("Ws: \n");
    // Print off Ws
    for (int l = 0; l < n_layers-1; l++) {
        printf("Layer %d\n", l);
        for (int n1 = 0; n1 < net_shape[l+1]; n1++) {
            printf("Col %d\n", n1);
            for (int n2 = 0; n2 < net_shape[l]; n2++) {
                printf("Row %d\n", n2);
                printf("Val: %f\n", Ws_c[l][n1][n2]);
            }
        }
    }
    printf("Fin_c:\n");
    for (int n = 0; n < net_shape[0]; n++) {
        printf("Neuron %d:\n", n);
        for (int f = 0; f < f_count_in[n]; f++) {
            printf("Firing time %d: %f\n", n, Fin_c[n][f]);
        }
    }
    printf("Scalar things:");
    printf("n_layers: %d", n_layers);
    printf("t_steps: %d", t_steps);
    printf("t_eps: %f", t_eps);

    // Do SRM0 simulation
    double **Fout;
    //int **f_count = (int **)calloc(net_shape.size(), sizeof(int *));
    int **f_count;
    cudaMallocManaged(&f_count, n_layers * sizeof(int *));
    Fout = par_sim_body_c(net_shape, n_layers, Fin_c, 
            f_count_in, f_max_c, Ws_c, f_count, t_steps, t_eps);

    // Print out the results
    //for (int l = 0; l < n_layers; l++) {
    //    cout << "Layer:" << l << endl;
    //    for (int n = 0; n < net_shape[l]; n++) {
    //        for (int f = 0; f < f_count[l][n]; f++) {
    //            cout << "Value:" << endl;
    //            cout << Fcal[l][n][f] << endl;
    //        }
    //    }
    //}
    

    //for (int l = 0; l < n_layers; l++) {
    //    for (int n = 0; n < net_shape[l]; n++) {
    //        cout << f_max[l][n] << endl;
    //    }
    //}

    // Print out fire counts
    for (int n = 0; n < net_shape[n_layers-1]; n++) {
        printf("Output Neuron %d had %d firing events\n", n, f_count[n_layers-1][n]);
    }

    // Pack up the last layer's firing times.
    int *cum_fires = (int *)calloc(net_shape[n_layers-1], sizeof(int));
    for (int n = 0; n < net_shape[n_layers-1]; n++) {
        if (n == 0) {
            cum_fires[n] = 0;
        } else {
            cum_fires[n] = cum_fires[n-1] + f_max[n_layers-1][n-1];
        }
    }

    //Flast = (double *)calloc(cum_fires[net_shape[n_layers-1]], sizeof(double));
    for (int n = 0; n < net_shape[n_layers-1]; n++) {
        for (int f = 0; f < f_max[n_layers-1][n]; f++) {
            printf("Assigning value %f\n", Fout[n][f]);
            Flast[f + cum_fires[n]] = Fout[n][f];
        }
    }

    //TODO: free things at some point.
}
