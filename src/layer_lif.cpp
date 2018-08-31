/** src/layer_lif.cpp Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 08.30.2018  */

#include <iostream>
#include <RcppArmadillo.h>
#include <math.h>       /* exp */
using namespace Rcpp;

// These should eventually be specifiable from R
double TAU = 1;
double LEAK = 0.5;
double V_THRESH = 1.5;

// [[Rcpp::depends(RcppArmadillo)]]


double kern(double dt) {
    if (dt <= 0) {
        return(0);
    }
    return(exp(-dt / TAU));
}
double kernd(double dt) {
    if (dt <= 0) {
        return(0);
    }
    return(-1/TAU * exp(-dt / TAU));
}

std::vector<std::vector<std::vector<double> > > lif_forward(int n_in, 
        int l_h, int* n_h, int n_out, std::vector<arma::mat> Ws, 
        std::vector<std::vector<double> > Fin, 
        double t_eps, int t_steps) {
    // n_h is of length l_h
    // Ws will be of length l_h + 2
    // the first element will be of length n_in, the second length given by the
    // first element of l_h, the third length by the second element of l_h, etc, and
    // finally the last length is given by n_out.
    //
    // Fin is of length n_out, as is flen. flen's elements give the length of each of 
    // Fin's elements
    //std::cout << "Here is W:\n";
    //for (int i = 0; i < layers; i++) {
    //    std::cout << "Layer " << i+1 << "\n";
    //}
    //

    // Store some info about the network
    int layers = 2 + l_h;//Includes input layer
    std::vector<int> net_shape(layers);
    net_shape[0] = n_in;
    for (int l = 0; l < l_h; l++) {
        net_shape[l+1] = *(n_h + l);
    }
    net_shape[layers-1] = n_out;

    // Initialize potential storage
    std::vector<arma::mat> Vs(layers-1);
    for (int i = 0; i < l_h; i++) {
        Vs[i] = arma::mat(*(n_h+i), t_steps + 1, arma::fill::zeros);
    }
    Vs[layers-2] = arma::mat(n_out, t_steps + 1, arma::fill::zeros);

    // Initilize totoal postsynaptic contribution
    // TODO: don't need this for output layer
    std::vector<arma::mat> ALPHA(layers);
    ALPHA[0] = arma::mat(n_in, t_steps, arma::fill::zeros);
    for (int i = 0; i < l_h; i++) {
        ALPHA[i+1] = arma::mat(*(n_h+i), t_steps, arma::fill::zeros);
    }
    ALPHA[layers-1] = arma::mat(n_out, t_steps, arma::fill::zeros);

    // Derivative of PSC
    std::vector<arma::mat> ALPHAd(layers);
    ALPHAd[0] = arma::mat(n_in, t_steps, arma::fill::zeros);
    for (int i = 0; i < l_h; i++) {
        ALPHAd[i+1] = arma::mat(*(n_h+i), t_steps, arma::fill::zeros);
    }
    ALPHAd[layers-1] = arma::mat(n_out, t_steps, arma::fill::zeros);

    // Storage for firing times
    std::vector<std::vector<std::vector<double> > > Fcal(layers);
    Fcal[0] = Fin;
    for (int l = 0; l < l_h; l++) {
        std::vector<std::vector<double> > Fi(*(n_h + l));
        for (int n = 0; n < *(n_h + l); n++) {
            Fi[n] = std::vector<double>();
        }
        Fcal[l+1] = Fi;
    }
    std::vector<std::vector<double> > Fi(n_out);
    for (int n = 0; n < n_out; n++) {
        Fi[n] = std::vector<double>();
    }
    Fcal[layers-1] = Fi;

    //// Firing times, those of the input layer are already known.
    //double*** Fcal = (double***) malloc(layers * sizeof(double**));
    //*(Fcal) = Fin;
    //// Storage for the hidden layers
    //double** Fcali;
    //for (int l = 0; l < l_h; l++) {
    //    Fcali = 
    //}
    //

    // Do simulation
    double t = 0;
    for (int ti = 0; ti < t_steps; ti++) {

        // Calculate total postsynaptic contribution and its derivative.
        for (int l = 0; l < layers; l++) {
            for (int n = 0; n < net_shape[l]; n++) {
                int n_f = Fcal[l][n].size();
                double psc = 0;
                double pscd = 0;
                for (int tfi = 0; tfi < n_f; tfi++) {
                    double tf = Fcal[l][n][tfi];
                    psc += kern(t - tf);
                    pscd += kernd(t - tf);
                }
                ALPHA[l](n, ti) = psc;
                ALPHAd[l](n, ti) = pscd;
            }
        }

        // Print the current ALPHA
        //std::cout << ti;
        //for (int l = 0; l < layers; l++) {
        //    std::cout << ALPHA[l].col(ti);
        //}

        // Multiply by weights to get input currents
        std::vector<arma::vec> h_inputs(l_h);
        if (l_h > 0) {
            for (int l = 0; l < l_h; l++) {
                h_inputs[l] = Ws[l].t() * ALPHA[l].col(ti);
            }
        }
        arma::vec out_input = Ws[l_h].t() * ALPHA[l_h].col(ti);

        // Get the gradient by factoring in the leaky term
        std::vector<arma::vec> h_dvdt(l_h);
        if (l_h > 0) {
            for (int l = 0; l < l_h; l++) {
                h_dvdt[l] = -LEAK * Vs[l].col(ti) + h_inputs[l];
            }
        }
        arma::vec out_dvdt = -LEAK * Vs[l_h].col(ti) + out_input;

        // Update the potentials
        if (l_h > 0) {
            for (int l = 0; l < l_h; l++) {
                Vs[l].col(ti+1) = Vs[l].col(ti) + t_eps * h_dvdt[l];
            }
        }
        Vs[l_h].col(ti+1) = Vs[l_h].col(ti) + t_eps * out_dvdt;

        // Check for firing neurons
        for (int l = 0; l < l_h+1; l++) {
            for (int n = 0; n < net_shape[l+1]; n++) {
                if (Vs[l](n, ti+1) > V_THRESH) {
                    Vs[l](n, ti+1) = 0;
                    Fcal[l+1][n].push_back(t + t_eps);
                }
            }
        }

        //
        // Print output
        //for (int l = 0; l < layers; l++) {
        //    std::cout << ALPHA[l] << "\n";
        //}
        //for (int l = 0; l < l_h; l++) {
        //    std::cout << h_inputs[l] << "\n";
        //}
        //std::cout << out_input << "\n";
        //for (int l = 0; l < l_h; l++) {
        //    std::cout << h_dvdt[l] << "\n";
        //}
        //std::cout << out_dvdt << "\n";
        //std::cout << ti << "\n";
        //for (int l = 0; l < l_h; l++) {
        //    std::cout << Vs[l] << "\n";
        //}
        //std::cout << Vs[l_h] << "\n";
        //std::cout << "\n";

        t += t_eps;
    }

    return(Fcal);
}

//' Solve the LIF forward problem
//'
//' Integrate a feedforward Linear Leaky Integrate and Fire Network via forward Euler.
//'
//' @param n_in The number of input neurons, a scalar integer.
//' @param l_h The number of hidden layers, a scalar integer.
//' @param n_h The number of hidden neurons in each layer (of length l_h).
//' @param n_out The number of output neurons, a scalar integer, for now has to be 1.
//' @param Ws A list of matrices giving the weights between layers, of length 1 fewer than the number of layers. The first matrix is of dimension n_in x n_h[1], etc.
//' @param Fin A list of numeric vectors, giving the times of the input spikes.
//' @param t_eps A scalar double, the step size for numerical integration.
//' @param t_steps A scalar integer, the number of finite difference iterations.
//' @return A list of lists of numeric vectors. The first entry will be Fin, and so on.
//' @export
// [[Rcpp::export]]
List goc(int n_in, int l_h, IntegerVector n_h, int n_out, List Ws, 
        List Fin, double t_eps, int t_steps) {

    // Build Ws matrices, put them into a Cpp list of arma matrices
    std::vector<arma::mat> Wsc(l_h+1);
    for (int i = 0; i < l_h+1; i++) {
        Wsc[i] = as<arma::mat>(Ws[i]);
    }

    // Translate Fin
    //double** Finc = (double**) malloc(n_in * sizeof(double*));
    //NumericVector Fi;
    //for (int i = 0; i < n_in; i++) {
    //    Fi = as<NumericVector>(Fin[i]);
    //    *(Finc + i) = Fi.begin();
    //}

    std::vector<std::vector<double> > Finc(n_in);
    for (int i = 0; i < n_in; i++) {
        Finc[i] = as<std::vector<double> >(Fin[i]);
    }

    //for (int i = 0; i < n_in; i++) {
    //    for (int j = 0; j < flen[i]; j++) {
    //        std::cout << *(*(Finc + i) + j) << "\n";
    //    }
    //}
    std::vector<std::vector<std::vector<double> > > Fcal;
    Fcal = lif_forward(n_in, l_h, n_h.begin(), n_out, 
            Wsc, Finc, t_eps, t_steps);

    List FcalR(l_h+2);
    for (int l = 0; l < l_h+2; l++) {
        FcalR[l] = Fcal[l];
    }

    return(FcalR);
}
