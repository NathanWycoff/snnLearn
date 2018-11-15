#include <iostream>
#include <R.h>
#include <Rinternals.h>

extern "C" void cool();

void cool() {
    double **a;
    cudaMallocManaged(&a, 10 * sizeof(double **));

    int n_layers = 5;
    int *net_shape = (int *)malloc(5 * sizeof(int));
    net_shape[0] = 2;
    net_shape[1] = 1;
    net_shape[2] = 10;
    net_shape[3] = 2;
    net_shape[4] = 5;

    int t_steps = 35;

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

}
