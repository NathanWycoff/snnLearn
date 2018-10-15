#!/usr/bin/Rscript
#  dev/arb_lif.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 10.15.2018

learn_rate <- 0.1
I_0 <- 1 #Stimulation
leak <- 0.5
tau <- 1
n_neurons <- 3
kern <- function(dt) as.numeric(dt>0) * exp(-dt/tau)
kernd <- function(dt) as.numeric(dt>0) * -1/tau * exp(-dt/tau)

t_eps <- 0.1
t_end <- 3.5
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

v_thresh <- 1.5
v_reset <- 0

N <- 2#Number of simulated neurons
M <- 3#Number of "input" neurons (firing times known)
OMEGA <- matrix(rnorm((N)^2), nrow = N)
LAMBDA <- matrix(rgamma(N*M,1,1), nrow = M)

nf <- 2# Number of times each input neuron fires during sim
Fin <- lapply(1:M, function(i) runif(nf, 0, t_end))

arb_forward_R <- function(N, M, OMEGA, LAMBDA, Fin, t_eps, t_steps) {
    # Initialize potential, postsynaptic potential, and firing times storage.
    V <- matrix(NA, nrow = N, ncol = t_steps+1)
    V[,1] <- 0
    ALPHA <- matrix(NA, nrow = N, ncol = t_steps)
    BETA <- matrix(NA, nrow = M, ncol = t_steps)
    Fcal <- lapply(1:N, function(n) list())

    t <- 0
    for (ti in 1:length(ts)) {
        # Update Potential
        ALPHA[,ti] <- sapply(Fcal, function(Fcali) 
                             sum(as.numeric(sapply(Fcali, function(tf) kern(t-tf)))))
        BETA[,ti] <- sapply(Fin, function(Fini) 
                             sum(as.numeric(sapply(Fini, function(tf) kern(t-tf)))))
        inputs <- t(OMEGA) %*% ALPHA[,ti] + t(LAMBDA) %*% BETA[,ti]
        dvdt <- -leak * V[,ti] + inputs
        V[,ti+1] <- V[,ti] + t_eps * dvdt

        # Check for firing events
        for (n in 1:N) {
            if (V[n,ti+1] > v_thresh) {
                Fcal[[n]] <- c(Fcal[[n]], t + t_eps)
            }
        }

        t <- t + t_eps
        
    }

    return(list(Fcal, ALPHA, BETA))
}

b <- arb_forward_R(N, M, OMEGA, LAMBDA, Fin, t_eps, t_steps)
