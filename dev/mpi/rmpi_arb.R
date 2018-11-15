#!/usr/bin/Rscript
#  rmpi_arb.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 10.15.2018

require('Rmpi')

mpi.spawn.Rslaves(nslaves = 4)

# Rank 0 partitions the work and sends it out to everyone.
output <- mpi.remote.exec({
    source('dev/rmpi_arb_setup.R')

    # Set up network params
    set.seed(123)
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

    N <- 10#Number of simulated neurons
    M <- 3#Number of "input" neurons (firing times known)
    OMEGA <- matrix(rnorm((N)^2), nrow = N)
    LAMBDA <- matrix(rgamma(N*M,1,1), nrow = M)

    nf <- 2# Number of times each input neuron fires during sim
    Fin <- lapply(1:M, function(i) runif(nf, 0, t_end))

    v_thresh <- 1.5
    leak <- 0.5

    proc <- naive_split_snn(mpi.comm.size()-1, N, M, OMEGA, LAMBDA, Fin, t_eps, t_steps)

    # Initialize potential, postsynaptic potential, and firing times storage.
    N_me <- length(proc$neurons)
    V <- matrix(NA, nrow = N_me, ncol = t_steps+1)
    V[,1] <- 0
    ALPHA <- matrix(NA, nrow = N, ncol = t_steps)
    BETA <- matrix(NA, nrow = M, ncol = t_steps)
    Fcal <- lapply(1:N, function(n) c())

    t <- 0
    for (ti in 1:length(ts)) {
        # Update Potential
        ALPHA[,ti] <- sapply(Fcal, function(Fcali) 
                             sum(as.numeric(sapply(Fcali, function(tf) kern(t-tf)))))
        BETA[,ti] <- sapply(Fin, function(Fini) 
                             sum(as.numeric(sapply(Fini, function(tf) kern(t-tf)))))
        inputs <- t(proc$OMEGA) %*% ALPHA[,ti] + t(proc$LAMBDA) %*% BETA[,ti]
        dvdt <- -leak * V[,ti] + inputs
        V[,ti+1] <- V[,ti] + t_eps * dvdt

        # Check for firing events
        Fcal_me <- c()
        for (n_i in 1:N_me) {
            n <- as.character(proc$neurons[n_i])
            if (V[n_i,ti+1] > v_thresh) {
                Fcal_me <- c(n, t + t_eps)
                V[n_i,ti+1] <- 0
            }
        }

        # Update sim time
        t <- t + t_eps

        # Communicate firing events
        # This will be an all to all in C
        for (n in 1:(mpi.comm.size()-1)) {
            mpi.isend.Robj(Fcal_me, n, tag = 42069)
        }

        # Listen to upstream neurons about their firing events.
        for (n in 1:(mpi.comm.size()-1)) {
            recv <- as.numeric(mpi.recv.Robj(n, tag = 42069))
            if (length(recv) > 0) {
                for (i in seq(1, length(recv)-1, by = 2)) {
                    Fcal[[recv[i]]] <- c(Fcal[[recv[i]]], recv[i+1])
                }
            }
        }
    }

    Fcal
})
output[[2]]

lapply(output, function(proc) proc)

mpi.quit()
