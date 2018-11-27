/** src/read_weights.c Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 10.26.2018  */
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
#define t_eps 0.001
#define THREADS_PER_BLOCK 512


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
void par_c_main_loop(double ***Vs, double ***ALPHA, double ***OMEGA, double **Fcal_l, int **f_count, double ***Ws, int* net_shape, int n_layers, 
        int t_steps, int l) {
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
                double tf = Fcal_l[n][tfi];
                psc += ipostkern(t - tf);
            }
            ALPHA[l][ti][n] = psc;

            if (l > 0) {
                // Update refractory contribution
                n_f = f_count[l][n];
                double ref = 0;
                for (int tfi = 0; tfi < n_f; tfi++) {
                    double tf = Fcal_l[n][tfi];
                    ref += iprekern(t - tf);
                }
                OMEGA[l-1][n][ti] = ref;

                // Update potential
                Vs[l-1][n][ti+1] = inner_prod(Ws[l-1][n], ALPHA[l-1][ti], net_shape[l-1]) + OMEGA[l-1][n][ti];
                //printf("l = %d, n = %d, ti = %d", l, n, ti);
                //printf("Vsl = %d, n = %d, ti = %d", l, n, ti);

                // Check for firing neurons
                if (Vs[l-1][n][ti+1] > V_THRESH) {
                    Fcal_l[n][f_count[l][n]] = t + t_eps;
                    f_count[l][n]++;
                }
            }
            t += t_eps;
        }
    }
}

// The main simulation, using armadillo for matrix multiplication, and organized in such a way that we solve a sequence embarassingly parallelizable problems.
double **par_sim_body_c(int *net_shape, int n_layers,
        double **Fin, int *f_count_in, long long int **f_max, double ***Ws,
        int** f_count) {
    // Do simulation
    int t_steps = 3500;

    // Get the layer with the most neurons
    int max_neur = 0;
    for (int l = 0; l < n_layers; l++) {
        if (max_neur < net_shape[l]) {
            max_neur = net_shape[l];
        }
    }

    // Stores electric potential for each layer in row major order.
    //double ***Vs = (double ***)calloc(n_layers-1, sizeof(double**));
    double ***Vs;
    cudaMallocManaged(&Vs, (n_layers-1) * sizeof(double **));
    for (int i = 0; i < n_layers-1; i++) {
        double **Vsi;
        cudaMallocManaged(&Vsi, net_shape[i+1] * sizeof(double*));
        Vs[i] = Vsi;
        //Vs[i] = (double **)calloc(net_shape[i+1], sizeof(double*));
        for (int j = 0; j < net_shape[i+1]; j++) {
            double *Vsij;
            cudaMallocManaged(&Vsij, (t_steps+1) * sizeof(double*));
            Vsij[0] = 0;// Initialize voltage at resting potential, assumed 0.
            Vs[i][j] = Vsij;
            //Vs[i][j] = (double *)calloc(t_steps + 1, sizeof(double));
        }
    }

    printf("After Vs\n");

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

    printf("After Weights copy\n");

    // Copy network shape to unified memory
    int *u_net_shape;
    cudaMallocManaged(&u_net_shape, n_layers * sizeof(int));
    cudaMemcpy(u_net_shape, net_shape, n_layers * sizeof(int), cudaMemcpyDefault);

    // Run actual inference
    int n_blocks = max_neur / THREADS_PER_BLOCK;
    if (n_blocks == 0) {
        n_blocks = 1;
    }

    for (int l = 0; l < n_layers; l++) {
        par_c_main_loop<<<n_blocks, THREADS_PER_BLOCK>>>(Vs, ALPHA, OMEGA, u_Fcal[l], u_f_count, u_Ws, u_net_shape, n_layers, 
                t_steps, l);
        //par_c_main_loop<<<1, 1>>>(Vs, ALPHA, OMEGA, u_Fcal, u_f_count, u_Ws, u_net_shape, n_layers, 
        //        t_steps, l);
    }
    cudaDeviceSynchronize();

    printf("After main loop\n");

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
    double **Fout = (double **)malloc(net_shape[n_layers]*sizeof(double*));
    for (int n = 0; n < net_shape[n_layers-1]; n++) {
        Fout[n] = (double *)malloc(f_max[n_layers-1][n] * sizeof(double));
        cudaMemcpy(Fout[n], u_Fcal[n_layers-1][n], f_max[n_layers-1][n] * sizeof(double), cudaMemcpyDefault);
    }

    // Copy f_count to host memory
    for (int l = 0; l < n_layers; l++) {
        f_count[l] = (int *)malloc(net_shape[l] * sizeof(int));
        cudaMemcpy(f_count[l], u_f_count[l], net_shape[l] * sizeof(int), cudaMemcpyDefault);
    }

    //TODO: copy f_count

    return(Fout);
}

int main () {
    // Read in weight matrix and store as array.
    std::ifstream input_ws("weights_file.dat");
    
    //std::string line;
    //for (std::string line; getline(input_ws, line); ) {
    //    std::cout << line << std::endl;
    //}

    bool firstline = true;
    std::vector<int> net_shape; // Stores the size of each network layer.
    std::vector<double**> Ws;// Stores weights in col major format.
    for(std::string line; getline(input_ws, line); ) {
        // Store the network configuration
        if (firstline) {
            std::istringstream net_shape_ss(line);

            while(net_shape_ss) {
                std::string s;
                if (!getline(net_shape_ss, s, ',')) break;
                //net_shape.push_back(stoi(s));
                int x;
                sscanf(s.c_str(), "%d", &x);
                net_shape.push_back(x);
            }

            firstline = false;
        }else {
            // Get the row and column size
            int rows = (int)std::count(line.begin(), line.end(), ';') + 1;
            double **W = (double **)malloc(rows * sizeof(double *));
            std::stringstream line_stream(line);
            std::string row_string;
            int r = 0;
            while(std::getline(line_stream, row_string, ';')) {
                int rows = (int)std::count(row_string.begin(), row_string.end(), ' ') + 1;
                double *col = (double *)malloc(rows * sizeof(double));
                std::stringstream elem_stream(row_string);
                std::string elem;
                int e = 0;
                while (std::getline(elem_stream, elem, ' ')) {
                    col[e] = std::stod(elem);
                    e++;
                }
                W[r] = col;
                r++;
            }
            Ws.push_back(W);
        }
    }

    int n_layers = net_shape.size();
    //// Look through the Ws
    //printf("Ws:");
    //for (int l = 0; l < n_layers-1; l++) {
    //    for (int n1 = 0; n1 < net_shape[l+1]; n1++) {
    //        for (int n2 = 0; n2 < net_shape[l]; n2++) {
    //            std::cout << Ws[l][n1][n2] << std::endl;
    //        }
    //    }
    //}
    
    // Read in the firing times.
    std::ifstream input_ft("input_ap.dat");

    std::vector<std::vector<double> > Fin; // Stores the size of each network layer.
    for(std::string line; std::getline(input_ft, line); ) {
        // Store the network configuration
        std::istringstream firing_times_ss(line);

        std::vector<double> Fline;
        while(firing_times_ss) {
            std::string s;
            if (!getline(firing_times_ss, s, ' ')) break;
            //Fline.push_back(stod(s));
            double x;
            std::sscanf(s.c_str(), "%lf", &x);
            Fline.push_back(x);
        }

        Fin.push_back(Fline);
    }

    //std::cout << "The Fins:" << std::endl;
    //for (int n = 0; n < Fin.size(); n++) {
    //    std::cout << "New Neuron:" << std::endl;
    //    for (int f = 0; f < Fin[n].size(); f++) {
    //        std::cout << Fin[n][f] << std::endl;
    //    }
    //}

    std::ifstream input_fmax("f_max_file.dat");
    long long int **f_max = (long long int **)malloc(n_layers * sizeof(long long int*));
    int l = 0;
    for(std::string line; std::getline(input_fmax, line, ';'); ) {
        f_max[l] = (long long int *)malloc(net_shape[l] * sizeof(long long int));
        std::stringstream line_stream(line);
        std::string neur_string;
        int n = 0;
        while(std::getline(line_stream, neur_string, ' ')) {
            f_max[l][n] = stoi(neur_string);
            n++;
        }
        l++;
    }

    // Convert input firing times to C array
    double **Fin_c = (double**)calloc(net_shape[0], sizeof(double *));
    int *f_count_in = (int *)calloc(net_shape[0], sizeof(int));
    for (int n = 0; n < net_shape[0]; n++) {
        f_count_in[n] = Fin[n].size();
        if (Fin[n].size() > 0) {
            Fin_c[n] = (double *)&Fin[n][0];
        }
    }

    // Convert Connection weights to a C array
    double ***Ws_c = (double***)calloc(net_shape.size()-1, sizeof(double**));
    for (int l = 0; l < net_shape.size()-1; l++) {
        Ws_c[l] = Ws[l];
    }

    // See if we got input firing times right
    //std::cout << "input firing times" << std::endl;
    //for (int n = 0; n < net_shape[0]; n++) {
    //    std::cout << f_count_in[n] << std::endl;
    //}

    // Do SRM0 simulation
    double **Fout;
    int **f_count = (int **)calloc(net_shape.size(), sizeof(int *));
    Fout = par_sim_body_c(&net_shape[0], net_shape.size(), Fin_c, 
            f_count_in, f_max, Ws_c, f_count);

    // Print out the results
    //for (int l = 0; l < net_shape.size(); l++) {
    //    std::cout << "Layer:" << l << std::endl;
    //    for (int n = 0; n < net_shape[l]; n++) {
    //        for (int f = 0; f < f_count[l][n]; f++) {
    //            std::cout << "Value:" << std::endl;
    //            std::cout << Fcal[l][n][f] << std::endl;
    //        }
    //    }
    //}
    
    // Print out fire counts
    for (int n = 0; n < net_shape[n_layers-1]; n++) {
        std::cout << "Output Neuron " << n << " had " << f_count[n_layers-1][n] << " firing events" << std::endl;
    }

    //TODO: free things at some point.

    return 0;
}
