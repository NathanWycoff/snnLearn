#!/usr/bin/Rscript
#  dev/mpi_integrate_lif.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.30.2018
kern <- function(dt) as.numeric(dt>0) * exp(-dt/tau)
kernd <- function(dt) as.numeric(dt>0) * -1/tau * exp(-dt/tau)
tau <- 1

# Create storage for the ODE integration
mpifun_setup_int <- function(proc, sizes, t_steps) {
    # Unpack sizes
    n_in <- sizes[1]
    n_out <- sizes[length(sizes)]
    if (length(sizes) > 2) {
        n_h <- sizes[2:(length(sizes)-1)]
    } else {
        n_h <- c()
    }

    ## Initialize Voltage Storage
    proc$Vs <- lapply(1:length(proc$layers), function(l) matrix(NA, nrow = length(proc$neurons[[l]]), ncol = t_steps + 1))

    # Initialize with the reset voltage
    for (i in 1:length(proc$Vs)) {
        proc$Vs[[i]][,1] <- 0
    }

    # Initialize ALPHA for presynaptic layers
    proc$ALPHA <- lapply(proc$Fcal, function(Fcali) matrix(NA, nrow = length(Fcali), ncol = t_steps))
    proc$ALPHAd <- lapply(proc$Fcal, function(Fcali) matrix(NA, nrow = length(Fcali), ncol = t_steps))

    return(proc)
}

mpifun_odestep <- function(proc, t, leak, t_eps, v_thresh) {
    L <- length(proc$layers)
    # Calculate Post-synaptic Potential 
    for (li in 1:L) {
        proc$ALPHA[[li]][,ti] <- sapply(proc$Fcal[[li]], function(Fc) 
                                  sum(as.numeric(sapply(Fc, function(tf) kern(t - tf)))))
        proc$ALPHAd[[li]][,ti] <- sapply(proc$Fcal[[li]], function(Fc) 
                                  sum(as.numeric(sapply(Fc, function(tf) kernd(t - tf)))))
    }

    # Calculate inputs
    inputs <- lapply(1:L, function(l) t(proc$Ws[[l]]) %*% proc$ALPHA[[l]][,ti])

    # Calculate derivative
    dvdt <- lapply(1:L, function(l) -leak * proc$Vs[[l]][,ti] + inputs[[l]])
    
    # Update potentials
    for (l in 1:L) {
        proc$Vs[[l]][,ti+1] <- proc$Vs[[l]][,ti] + t_eps * dvdt[[l]]
    }

    # See if anyone fired and if so record it.
    post_Fcal <- lapply(1:L, function(l) lapply(1:length(proc$neurons[[l]]), function(n) list()))
    for (l in 1:L) {
        for (n in 1:nrow(proc$Vs[[l]])) {
            if (proc$Vs[[l]][n,ti+1] > v_thresh) {
                proc$Vs[[l]][n,ti+1] <- 0
                post_Fcal[[l]][[n]] <- c(post_Fcal[[l]][[n]], t + t_eps)
            }
        }
    }

    return(list(proc = proc, post_Fcal = post_Fcal))
}
