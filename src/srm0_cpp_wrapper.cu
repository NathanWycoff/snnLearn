/** src/read_weights.c Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 10.26.2018  */
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include "srm0.h"


// NOTE: Need to compile in C++11 mode, add -std=c++11
// These should eventually be specifiable from R
#define THREADS_PER_BLOCK 512

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
            int n_cols = (int)std::count(line.begin(), line.end(), ';') + 1;
            double **W = (double **)malloc(n_cols * sizeof(double *));
            std::stringstream line_stream(line);
            std::string col_string;
            int r = 0;
            while(std::getline(line_stream, col_string, ';')) {
                int n_rows = (int)std::count(col_string.begin(), col_string.end(), ' ') + 1;
                double *col = (double *)malloc(n_rows * sizeof(double));
                std::stringstream elem_stream(col_string);
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
    int t_steps = 3500;
    double t_eps = 0.001;
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

    double **Fout;
    int **f_count = (int **)calloc(net_shape.size(), sizeof(int *));
    Fout = par_sim_body_c(&net_shape[0], net_shape.size(), Fin_c, 
            f_count_in, f_max, Ws_c, f_count, t_steps, t_eps);

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
