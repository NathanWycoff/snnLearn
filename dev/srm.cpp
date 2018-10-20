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

    // Do SRM0 simulation
    int t_steps = 35;
    int n_h = net_shape.size()-2;

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
        std::vector<std::vector<double> > Fi(net_shape[l+1]);
        for (int n = 0; n < net_shape[l+1]; n++) {
            Fi[n] = std::vector<double>();
        }
        Fcal[l+1] = Fi;
    }

    // Do simulation
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

    // Print out the results
    for (int l = 0; l < Fcal.size(); l++) {
        cout << "Layer:" << l << endl;
        for (int n = 0; n < Fcal[l].size(); n++) {
            for (int f = 0; f < Fcal[l][n].size(); f++) {
                cout << "Value:" << endl;
                cout << Fcal[l][n][f] << endl;
            }
        }
    }

    return 0;
}
