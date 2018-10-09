#!/usr/bin/Rscript
#  R/layer_lif.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 08.28.2018

##' Solve the LIF forward problem in R
##'
##' Integrate a feedforward Linear Leaky Integrate and Fire Network via forward Euler.
##'
##' @param n_in The number of input neurons, a scalar integer.
##' @param l_h The number of hidden layers, a scalar integer.
##' @param n_h The number of hidden neurons in each layer (of length l_h).
##' @param n_out The number of output neurons, a scalar integer, for now has to be 1.
##' @param Ws A list of matrices giving the weights between layers, of length 1 fewer than the number of layers. The first matrix is of dimension n_in x n_h[1], etc.
##' @param Fin A list of numeric vectors, giving the times of the input spikes.
##' @param t_eps A scalar double, the step size for numerical integration.
##' @param t_steps A scalar integer, the number of finite difference iterations.
##' @return A list of lists of numeric vectors. The first entry will be Fin, and so on.
##' @examples
##' t_eps <- 0.01
##' t_end <- 10
##' ts <- seq(0, t_end, by = t_eps)
##' t_steps <- length(ts)
##' n_in <- 2
##' n_out <- 1
##' n_h <- c(1)
##' layers <- 2 + length(n_h)
##' #Ws <- list(matrix(c(3.5), ncol = 1), matrix(c(3), ncol = 1))
##' Fin <- list(seq(0,10, by = 1), seq(0,10, by = 1))
##' # Generate random wieghts
##' set.seed(123)
##' sizes <- c(n_in, n_h, n_out)
##' Ws <- lapply(1:(length(sizes)-1), function(i) 
##'              matrix(rnorm(sizes[i]*sizes[i+1], 3), nrow = sizes[i], ncol = sizes[i+1]))
##' l_h <- length(n_h)
##' a <- lif_forward_R(n_in, l_h, n_h, n_out, Ws, Fin, t_eps, t_steps) 
##' cat('Firing Times by layer-neuron:\n')
##' print(a)
##'
##' @export
## [[Rcpp::export]]
lif_forward_R <- function(n_in, l_h, n_h, n_out, Ws, Fin, t_eps, t_steps) {
    layers <- 2 + length(n_h)

    tau = 1;
    leak = 0.5;
    v_thresh = 1.5;
    kern <- function(dt) as.numeric(dt>0) * exp(-dt/tau)
    kernd <- function(dt) as.numeric(dt>0) * -1/tau * exp(-dt/tau)

    ## Initialize Voltage Storage
    Vs <- list()
    if (length(n_h) > 0) {
        for (i in 1:length(n_h)) {
            Vs[[i]] <- matrix(NA, nrow = n_h[i], ncol = t_steps + 1)
        }
    }
    Vs[[length(n_h) + 1]] <- matrix(NA, nrow = n_out, ncol = t_steps + 1)

    # Initialize with the reset voltage
    for (i in 1:length(Vs)) {
        Vs[[i]][,1] <- 0
    }

    ## Initialize Postsynaptic Potential Storage
    ALPHA <- list()
    ALPHA[[1]] <- matrix(NA, nrow = n_in, ncol = t_steps)
    if (length(n_h) > 0) {
        for (i in 1:length(n_h)) {
            ALPHA[[i+1]] <- matrix(NA, nrow = n_h[i], ncol = t_steps)
        }
    }
    ALPHA[[length(n_h) + 2]] <- matrix(NA, nrow = n_out, ncol = t_steps)

    # And its derivative too
    ALPHAd <- list()
    ALPHAd[[1]] <- matrix(NA, nrow = n_in, ncol = t_steps)
    if (length(n_h) > 0) {
        for (i in 1:length(n_h)) {
            ALPHAd[[i+1]] <- matrix(NA, nrow = n_h[i], ncol = t_steps)
        }
    }
    ALPHAd[[length(n_h) + 2]] <- matrix(NA, nrow = n_out, ncol = t_steps)

    ## Initialize storage for firing times
    Fcal <- list(Fin)
    if (length(n_h) > 0) {
        for (i in 1:length(n_h)) {
            Fcal[[i+1]] <- lapply(1:n_h[i], function(j) c())
        }
    }
    Fcal[[length(n_h)+2]] <- lapply(1:n_out, function(j) c())

    # Integrate ODE system using Forward Euler
    t <- 0
    for (ti in 1:t_steps) {
        ## Update the network, layer by layer

        # Calculate Post-synaptic Potential 
        for (l in 1:layers) {
            ALPHA[[l]][,ti] <- sapply(Fcal[[l]], function(Fc) 
                   sum(as.numeric(sapply(Fc, function(tf) kern(t - tf)))))
            ALPHAd[[l]][,ti] <- sapply(Fcal[[l]], function(Fc) 
                   sum(as.numeric(sapply(Fc, function(tf) kernd(t - tf)))))
        }

        # Calculate inputs
        if (length(n_h) > 0) {
            h_inputs <- lapply(1:length(n_h), function(l) 
                               t(Ws[[l]]) %*% ALPHA[[l]][,ti])
        }
        out_input <- t(Ws[[length(n_h)+1]]) %*% ALPHA[[length(n_h)+1]][,ti]

        # Calculate derivative
        if (length(n_h) > 0) {
            h_dvdt <- lapply(1:length(n_h), function(l) 
                             -leak * Vs[[l]][,ti] + h_inputs[[l]])
        }
        out_dvdt <- -leak * Vs[[length(n_h)+1]][,ti] + out_input


        # Update the potentials
        if (length(n_h) > 0) {
            for (l in 1:length(n_h)) {
                Vs[[l]][,ti+1] <- Vs[[l]][,ti] + t_eps * h_dvdt[[l]]
            }
        }
        Vs[[length(n_h)+1]][,ti+1] <- Vs[[length(n_h)+1]][,ti] + t_eps * out_dvdt


        for (l in 1:length(Vs)) {
            for (n in 1:nrow(Vs[[l]])) {
                if (Vs[[l]][n,ti+1] > v_thresh) {
                    Vs[[l]][n,ti+1] <- 0
                    Fcal[[l+1]][[n]] <- c(Fcal[[l+1]][[n]], t + t_eps)
                }
            }
        }

        t <- t + t_eps
    }
    #return(list(Fcal = Fcal, ALPHA=ALPHA, ALPHAd = ALPHAd))
    return(Fcal) 
}
