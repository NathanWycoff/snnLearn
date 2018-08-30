#!/usr/bin/Rscript
#  R/layer_lif.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 08.28.2018

## A LIF with layer structure
learn_rate <- 0.1
I_0 <- 1 #Stimulation
leak <- 0.5
tau <- 1
n_neurons <- 3
kern <- function(dt) as.numeric(dt>0) * exp(-dt/tau)
kernd <- function(dt) as.numeric(dt>0) * -1/tau * exp(-dt/tau)

t_eps <- 1
t_end <- 20
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

v_thresh <- 1.5
v_reset <- 0

n_in <- 2
n_out <- 1
n_h <- c(1)
layers <- 2 + length(n_h)
#Ws <- list(matrix(c(3.5), ncol = 1), matrix(c(3), ncol = 1))

Fin <- list(seq(0,10, by = 1), 10+seq(0,10, by = 1))

# Generate random wieghts
set.seed(123)
sizes <- c(n_in, n_h, n_out)
Ws <- lapply(1:(length(sizes)-1), function(i) 
             matrix(rnorm(sizes[i]*sizes[i+1], 3), nrow = sizes[i], ncol = sizes[i+1]))

gon <- function(Ws, Fin) {
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
    for (ti in 1:length(ts)) {
        t <- ts[ti]

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
    }
    return(list(Fcal, ALPHA, ALPHAd))
}

gon(Ws, Fin)[[1]]

td <- 12
iters <- 100
for (iter in 1:iters) {
    #Assumes at least 1 hidden layer
    ret <- gon(Ws, Fin)
    Fcal <- ret[[1]]
    ALPHA <- ret[[2]]
    ALPHAd <- ret[[3]]

    ta <- Fcal[[length(n_h) + 2]][[1]][1]
    print(ta)
    tai <- which(abs(ts-ta) < t_eps/2)
    # Output delta
    d_out <- rep(NA, n_out)
    for (neur in 1:n_out) {
        #TODO: one neuron assumption: modify td & ta
        d_out[neur] <- -(ta - td) / t(Ws[[length(n_h)+1]]) %*% ALPHAd[[length(n_h)+1]][,tai]
    }

    # Hidden Delta
    d_h <- lapply(n_h, function(h) rep(NA, h))
    for (l in length(n_h):1) {
        for (neur in 1:n_h[l]) {
            #TODO: d_out not good for more than 1 hidden layer
            d_h[[l]][neur] <-  d_out * (ALPHAd[[l+1]][neur,tai] * sum(Ws[[l+1]][neur,]))  /
                t(Ws[[l]][,neur]) %*% ALPHAd[[l]][,tai]
        }
    }
    delta <- d_h
    delta[[length(delta)+1]] <- d_out

    # Calculate weight updates, and apply them
    for (wi in 1:length(Ws)) {
        Wd <- -t(delta[[wi]]) %x% ALPHA[[wi]][,tai]
        #print(Wd)
        Ws[[wi]] <- Ws[[wi]] - learn_rate * Wd
    }
} 
