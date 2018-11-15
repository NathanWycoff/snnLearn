/** dev/srm.cpp Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 10.19.2018  */

// Solve an SRM0 model
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

// These should eventually be specifiable from R
double TAU = 1;
double V_THRESH = 1.5;
double t_eps = 0.1;

// Integrated Postsynaptic Kernel
double ipostkern(double dt) {
    if (dt < 0) {
        return(0);
    }
    return(TAU * (1 - exp(-dt / TAU)));
}

// Integrated refractory kernel.
double iprekern(double dt) {
    if (dt < 0) {
        return(0);
    }
    return(-V_THRESH);
}

// The inner product function, uses the standard R^n inner product.
double inner_prod(double *x, double *y, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }   
    return(sum);
}


// Row-major Matrix-vector multiplication
// A points to m many length n double pointers, each one of A's rows.
// x is, of course, of length n.
// We return a vector of length m.
//double* rm_mv_mult(double **A, int m, int n, double *x) {
//    double *b = (double *)calloc(m, sizeof(double));
//    for (int i = 0; i < m; i++) {
//        b[i] = inner_prod(A[i], x, n);
//    }
//    return(b);
//}

// The main simulation, using armadillo for matrix multiplication.
vector<vector<vector<double> > > sim_body_arma(vector<int> net_shape, 
        vector<vector<double> > Fin, vector<mat> Ws) {
    // Do simulation
    int t_steps = 35;

    vector<mat> Vs;
    for (int i = 0; i < net_shape.size()-1; i++) {
        Vs.push_back(mat(net_shape[i+1], t_steps + 1, fill::zeros));
    }

    vector<mat> ALPHA;
    vector<mat> OMEGA;
    for (int i = 0; i < net_shape.size(); i++) {
        ALPHA.push_back(mat(net_shape[i], t_steps, fill::zeros));
        OMEGA.push_back(mat(net_shape[i], t_steps, fill::zeros));
    }

    // Storage for firing times
    vector<vector<vector<double> > > Fcal(net_shape.size());
    Fcal[0] = Fin;
    for (int l = 0; l < net_shape.size()-1; l++) {
        vector<vector<double> > Fi(net_shape[l+1]);
        for (int n = 0; n < net_shape[l+1]; n++) {
            Fi[n] = vector<double>();
        }
        Fcal[l+1] = Fi;
    }

    double t = 0;
    for (int ti = 0; ti < t_steps; ti++) {

        // Calculate total postsynaptic and refractory contribution 
        for (int l = 0; l < net_shape.size(); l++) {
            for (int n = 0; n < net_shape[l]; n++) {
                int n_f = Fcal[l][n].size();
                double psc = 0;
                double ref = 0;
                for (int tfi = 0; tfi < n_f; tfi++) {
                    double tf = Fcal[l][n][tfi];
                    psc += ipostkern(t - tf);
                    ref += iprekern(t - tf);
                }
                ALPHA[l](n, ti) = psc;
                OMEGA[l](n, ti) = ref;
            }
        }

        // Update potential
        for (int l = 0; l < net_shape.size()-1; l++) {
            Vs[l].col(ti+1) = Ws[l].t() * ALPHA[l].col(ti) + OMEGA[l+1].col(ti);
        }

        // Check for firing neurons
        for (int l = 0; l < net_shape.size()-1; l++) {
            for (int n = 0; n < net_shape[l+1]; n++) {
                if (Vs[l](n, ti+1) > V_THRESH) {
                    Fcal[l+1][n].push_back(t + t_eps);
                }
            }
        }

        t += t_eps;
    }

    return(Fcal);
}

// The main simulation, using armadillo for matrix multiplication, and organized in such a way that we solve a sequence embarassingly parallelizable problems.
vector<vector<vector<double> > > par_sim_body_arma(vector<int> net_shape, 
        vector<vector<double> > Fin, vector<mat> Ws) {
    // Do simulation
    int t_steps = 35;

    vector<mat> Vs;
    for (int i = 0; i < net_shape.size()-1; i++) {
        Vs.push_back(mat(net_shape[i+1], t_steps + 1, fill::zeros));
    }

    vector<mat> ALPHA;
    vector<mat> OMEGA;
    for (int i = 0; i < net_shape.size(); i++) {
        ALPHA.push_back(mat(net_shape[i], t_steps, fill::zeros));
        if (i > 0) {
            OMEGA.push_back(mat(net_shape[i], t_steps, fill::zeros));
        }
    }

    // Storage for firing times
    vector<vector<vector<double> > > Fcal(net_shape.size());
    Fcal[0] = Fin;
    for (int l = 0; l < net_shape.size()-1; l++) {
        vector<vector<double> > Fi(net_shape[l+1]);
        for (int n = 0; n < net_shape[l+1]; n++) {
            Fi[n] = vector<double>();
        }
        Fcal[l+1] = Fi;
    }

    double t;
    for (int l = 0; l < net_shape.size(); l++) {
        for (int n = 0; n < net_shape[l]; n++) {
            t = 0;
            for (int ti = 0; ti < t_steps; ti++) {
                // Calculate total postsynaptic contribution 
                int n_f = Fcal[l][n].size();
                double psc = 0;
                for (int tfi = 0; tfi < n_f; tfi++) {
                    double tf = Fcal[l][n][tfi];
                    psc += ipostkern(t - tf);
                }
                ALPHA[l](n, ti) = psc;

                if (l > 0) {
                    // Update refractory contribution
                    int n_f = Fcal[l][n].size();
                    double ref = 0;
                    for (int tfi = 0; tfi < n_f; tfi++) {
                        double tf = Fcal[l][n][tfi];
                        ref += iprekern(t - tf);
                    }
                    OMEGA[l-1](n, ti) = ref;
                    
                    // Update potential
                    Vs[l-1](n, ti+1) = inner_prod(Ws[l-1].colptr(n), ALPHA[l-1].colptr(ti), net_shape[l-1]) + OMEGA[l-1](n, ti);

                    // Check for firing neurons
                    if (Vs[l-1](n, ti+1) > V_THRESH) {
                        Fcal[l][n].push_back(t + t_eps);
                    }
                }
                t += t_eps;
            }
        }
    }

    return(Fcal);
}

void par_c_main_loop(double ***Vs, double ***ALPHA, double ***OMEGA, double ***Fcal,
        int **f_count, double ***Ws, int* net_shape, int n_layers, 
        int t_steps, double t_eps) {
    double t;
    for (int l = 0; l < n_layers; l++) {
        for (int n = 0; n < net_shape[l]; n++) {
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
                        cout << f_count[l][n] << endl;
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

    // Stores electric potential for each layer in row major order.
    double ***Vs = (double ***)calloc(n_layers-1, sizeof(double**));
    for (int i = 0; i < n_layers-1; i++) {
        Vs[i] = (double **)calloc(net_shape[i+1], sizeof(double*));
        for (int j = 0; j < net_shape[i+1]; j++) {
            Vs[i][j] = (double *)calloc(t_steps + 1, sizeof(double));
        }
    }

    // ALPHA stores integrated postsynaptic potential in column major order.
    // OMEGA stores integrated refractory contribution in row major order.
    double ***ALPHA = (double ***)calloc(n_layers, sizeof(double**));
    double ***OMEGA = (double ***)calloc(n_layers-1, sizeof(double**));
    for (int i = 0; i < n_layers; i++) {
        ALPHA[i] = (double **) calloc(t_steps, sizeof(double*));
        for (int j = 0; j < t_steps; j++) {
            ALPHA[i][j] = (double *) calloc(net_shape[i], sizeof(double));
        }
        if (i > 0) {
            OMEGA[i-1] = (double **) calloc(net_shape[i], sizeof(double*));
            for (int j = 0; j < net_shape[i]; j++) {
                OMEGA[i-1][j] = (double *) calloc(t_steps, sizeof(double));
            }
        }
    }

    // Storage for firing times
    double ***Fcal = (double ***)calloc(n_layers, sizeof(double**));
    f_count[0] = f_count_in;
    Fcal[0] = Fin;
    for (int l = 0; l < n_layers-1; l++) {
        double **Fi = (double **) calloc(net_shape[l+1], sizeof(double *));
        f_count[l+1] = (int *)calloc(net_shape[l+1], sizeof(int));
        for (int n = 0; n < net_shape[l+1]; n++) {
            Fi[n] = (double *) calloc(f_max[l+1][n], sizeof(double));
        }
        Fcal[l+1] = Fi;
    }

    // Run actual inference
    par_c_main_loop(Vs, ALPHA, OMEGA, Fcal, f_count, Ws, net_shape, n_layers, 
            t_steps, t_eps);

    // Clean up
    for (int i = 0; i < n_layers-1; i++) {
        for (int j = 0; j < net_shape[i+1]; j++) {
            free(Vs[i][j]); 
        }
        free(Vs[i]);
    }
    free(Vs);

    // ALPHA stores integrated postsynaptic potential in column major order.
    // OMEGA stores integrated refractory contribution in row major order.
    for (int i = 0; i < n_layers; i++) {
        for (int j = 0; j < t_steps; j++) {
            free(ALPHA[i][j]);
        }
        free(ALPHA[i]);
        if (i > 0) {
            for (int j = 0; j < net_shape[i]; j++) {
                free(OMEGA[i-1][j]);
            }
            free(OMEGA[i-1]);
        }
    }
    free(ALPHA);
    free(OMEGA);

    return(Fcal);
}

int main () {
    // Read in weight matrix and store as array.
    ifstream input_ws("weights_file.dat");

    bool firstline = true;
    vector<int> net_shape; // Stores the size of each network layer.
    vector<mat> Ws; // Stores weights in column major format.
    for(string line; getline(input_ws, line); ) {
        // Store the network configuration
        if (firstline) {
            istringstream net_shape_ss(line);

            while(net_shape_ss) {
                string s;
                if (!getline(net_shape_ss, s, ',')) break;
                net_shape.push_back(stoi(s));
            }

            firstline = false;
        } else {
            Ws.push_back(mat(line));
        }
    }

    // Read in the firing times.
    ifstream input_ft("input_ap.dat");

    vector<vector<double> > Fin; // Stores the size of each network layer.
    for(string line; getline(input_ft, line); ) {
        // Store the network configuration
        istringstream firing_times_ss(line);

        vector<double> Fline;
        while(firing_times_ss) {
            string s;
            if (!getline(firing_times_ss, s, ' ')) break;
            Fline.push_back(stod(s));
        }

        Fin.push_back(Fline);
    }

    // Calculate an upper bound on how many times each neuron will fire.
    //int ** f_max = (int **)calloc(net_shape.size(), sizeof(int*));
    vector<ivec> f_max(net_shape.size());
    f_max[0] = ivec(net_shape[0]);
    for (int n = 0; n < net_shape[0]; n++) {
        f_max[0][n] = Fin[n].size();
    }
    for (int l = 0; l < net_shape.size()-1; l++) {
        f_max[l+1] = conv_to<ivec>::from(TAU / V_THRESH * Ws[l].t() * f_max[l]);
    }

    // Convert to a C array for use with GPU
    long long int **f_max_c = 
        (long long int **)calloc(net_shape.size(), sizeof(long long int*));
    for (int l = 0; l < net_shape.size(); l++) {
        f_max_c[l] = f_max[l].memptr();
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
        Ws_c[l] = (double**)calloc(net_shape[l], sizeof(double*));
        for (int n = 0; n < net_shape[l]; n++) {
            Ws_c[l][n] = Ws[l].colptr(n);
        }
    }

    // Do SRM0 simulation
    double ***Fcal;
    int **f_count = (int **)calloc(net_shape.size(), sizeof(int *));
    Fcal = par_sim_body_c(&net_shape[0], net_shape.size(), Fin_c, 
            f_count_in, f_max_c, Ws_c, f_count);

    // Print out the results
    for (int l = 0; l < net_shape.size(); l++) {
        cout << "Layer:" << l << endl;
        for (int n = 0; n < net_shape[l]; n++) {
            for (int f = 0; f < f_count[l][n]; f++) {
                cout << "Value:" << endl;
                cout << Fcal[l][n][f] << endl;
            }
        }
    }
    

    for (int l = 0; l < net_shape.size(); l++) {
        cout << f_max[l] << endl;
    }

    //TODO: free things at some point.

    return 0;
}
