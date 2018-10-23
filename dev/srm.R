#!/usr/bin/Rscript
#  dev/srm.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 10.19.2018

## The Spike Response Model
I_0 <- 1 #Stimulation
tau <- 1
n_neurons <- 3
ipostkern <- function(dt) as.numeric(dt>=0) * tau * (1 - exp(-dt/tau))# Integrated kernel
iprekern <- function(dt) as.numeric(dt>=0) * -v_thresh# Integrated kernel


t_eps <- 0.1
t_end <- 3.5
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

v_thresh <- 1.5
v_reset <- 0

n_in <- 2
n_out <- 1
n_h <- c(1)
layers <- 2 + length(n_h)
#Ws <- list(matrix(c(3.5), ncol = 1), matrix(c(3), ncol = 1))

Fin <- list(seq(0,1, by = 1), seq(0,1, by = 1))

# Generate random wieghts
set.seed(123)
sizes <- c(n_in, n_h, n_out)
Ws <- lapply(1:(length(sizes)-1), function(i) 
             matrix(rnorm(sizes[i]*sizes[i+1], 3), nrow = sizes[i], ncol = sizes[i+1]))

# Calculate an upper bound on how many times a neuron will fire.
predict_fire_counts <- function(Ws, Fin) {
    counts <- list()
    last_count <- sapply(Fin, length)
    counts[[1]] <- last_count
    for (l in 1:length(Ws)) {
        counts[[l+1]] <- floor(tau / v_thresh * t(Ws[[l]]) %*% counts[[l]])
    }
    return(counts)
}

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

    ## Initialize Postsynaptic integrated kernel Storage
    ALPHA <- list()
    ALPHA[[1]] <- matrix(NA, nrow = n_in, ncol = t_steps)
    if (length(n_h) > 0) {
        for (i in 1:length(n_h)) {
            ALPHA[[i+1]] <- matrix(NA, nrow = n_h[i], ncol = t_steps)
        }
    }
    ALPHA[[length(n_h) + 2]] <- matrix(NA, nrow = n_out, ncol = t_steps)

    ## Initialize integrated refractory kernel
    OMEGA <- list()
    OMEGA[[1]] <- matrix(NA, nrow = n_in, ncol = t_steps)
    if (length(n_h) > 0) {
        for (i in 1:length(n_h)) {
            OMEGA[[i+1]] <- matrix(NA, nrow = n_h[i], ncol = t_steps)
        }
    }
    OMEGA[[length(n_h) + 2]] <- matrix(NA, nrow = n_out, ncol = t_steps)

    ## Initialize storage for firing times
    Fcal <- list(Fin)
    if (length(n_h) > 0) {
        for (i in 1:length(n_h)) {
            Fcal[[i+1]] <- lapply(1:n_h[i], function(j) c())
        }
    }
    Fcal[[length(n_h)+2]] <- lapply(1:n_out, function(j) c())

    # Apply filters
    t <- 0
    for (ti in 1:length(ts)) {
        ## Update the network, layer by layer

        # Calculate Post-synaptic Potential 
        for (l in 1:layers) {
            ALPHA[[l]][,ti] <- sapply(Fcal[[l]], function(Fc) 
                   sum(as.numeric(sapply(Fc, function(tf) ipostkern(t - tf)))))
            OMEGA[[l]][,ti] <- sapply(Fcal[[l]], function(Fc) 
                   sum(as.numeric(sapply(Fc, function(tf) iprekern(t - tf)))))
        }

        # Calculate inputs
        if (length(n_h) > 0) {
            for (l in 1:n_h) {
                Vs[[l]][,ti+1] <- t(Ws[[l]]) %*% ALPHA[[l]][,ti] + OMEGA[[l+1]][,ti]
            }
        }
        Vs[[length(n_h) + 1]][,ti+1] <- 
            t(Ws[[length(n_h)+1]]) %*% ALPHA[[length(n_h)+1]][,ti] + OMEGA[[length(n_h)+2]][,ti]

        # Detect Firing events
        for (l in 1:length(Vs)) {
            for (n in 1:nrow(Vs[[l]])) {
                if (Vs[[l]][n,ti+1] > v_thresh) {
                    Fcal[[l+1]][[n]] <- c(Fcal[[l+1]][[n]], t + t_eps)
                }
            }
        }

        t <- t + t_eps
    }
    return(list(Fcal, Vs))
}

b <- gon(Ws, Fin)
