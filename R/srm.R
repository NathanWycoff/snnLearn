#!/usr/bin/Rscript
#  R/srm.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.25.2018

source('R/lib.R')

srm0_R <- function(Ws, net_shape, Fin, t_steps, t_eps, tau = 1, v_thresh = 1.5, debug = 0, copy_gamma = FALSE) {
    ts <- seq(0, t_eps * (t_steps-1), by = t_eps)
    ipostkern <- function(dt) as.numeric(dt>=0) * tau * (1 - exp(-dt/tau))# Integrated kernel
    postkern <- function(dt) as.numeric(dt>=0) * exp(-dt/tau)# Simply the kernel itself
    dpostkern <- function(dt) as.numeric(dt>=0) * (-1)/tau * exp(-dt/tau)# Derivative of the kernel 
    iprekern <- function(dt) as.numeric(dt>=0) * -v_thresh# Integrated kernel

    f_max <- predict_fire_counts(Ws, Fin)

    # Unpack some stuff
    n_layers <- length(net_shape)
    n_in <- net_shape[1]
    if (n_layers > 2) {
        n_h <- net_shape[2:(n_layers-1)]
    }
    n_out <- net_shape[n_layers]

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

    ## Initialize synaptic contribution storage
    if (copy_gamma) {
        GAMMA <- lapply(1:n_out, function(i) list())
        GAMMAd <- lapply(1:n_out, function(i) list())
    }

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
        for (l in 1:n_layers) {
            ALPHA[[l]][,ti] <- sapply(Fcal[[l]], function(Fc) 
                   sum(as.numeric(sapply(Fc, function(tf) ipostkern(t - tf)))))
            OMEGA[[l]][,ti] <- sapply(Fcal[[l]], function(Fc) 
                   sum(as.numeric(sapply(Fc, function(tf) iprekern(t - tf)))))
        }

        # Calculate instantaneous contribution
        #for (l in 1:n_layers) {
        #    GAMMA[[l]][,ti] <- sapply(Fcal[[l]], function(Fc) sum(as.numeric(sapply(Fc, function(tf) postkern(t - tf)))))
        #    GAMMAd[[l]][,ti] <- sapply(Fcal[[l]], function(Fc) sum(as.numeric(sapply(Fc, function(tf) dpostkern(t - tf)))))
        #}

        # Calculate inputs
        if (length(n_h) > 0) {
            for (l in 1:length(n_h)) {
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
                    # Record additional information if its an output neuron
                    if (copy_gamma && l == n_layers-1) {
                        GAMMA[[n]][[length(GAMMA[[n]])+1]] <- lapply(1:n_layers, function(l) 
                                                                   sapply(Fcal[[l]], function(Fc) sum(as.numeric(sapply(Fc, function(tf) postkern(t + t_eps - tf))))))
                        GAMMAd[[n]][[length(GAMMAd[[n]])+1]] <- lapply(1:n_layers, function(l) 
                                                                   sapply(Fcal[[l]], function(Fc) sum(as.numeric(sapply(Fc, function(tf) dpostkern(t + t_eps - tf))))))
                    }
                }
            }
        }

        t <- t + t_eps
    }
    if (copy_gamma) {
        return(list(Fout = Fcal[[n_layers]], GAMMA = GAMMA, GAMMAd = GAMMAd))
    } else {
        return(list(Fout = Fcal[[n_layers]]))
    }
}

#' Wrapper for the SRM0 model written in cuda/C
srm0_cu <- function(Ws, net_shape, Fin, t_steps, t_eps, debug_in = 0, copy_gamma_in = FALSE) {

    # Calculate maximum firing times for each neuron
    f_max <- predict_fire_counts(Ws, Fin)

    # Prepare inputs for C
    n_layers <- length(net_shape)
    w <- unlist(Ws)
    c <- rep(0, n_layers)
    Finc <- unlist(Fin)
    f_count_in <- sapply(Fin, length)
    f_max_R <- unlist(f_max)
    Flast <- rep(-1, sum(f_max[[length(f_max)]]))
    gamma <- rep(-1, sum(net_shape) * sum(f_max[[length(f_max)]]))
    gammad <- rep(-1, sum(net_shape) * sum(f_max[[length(f_max)]]))
    debug <- c(debug_in)
    copy_gamma <- c(copy_gamma_in)

    rst <- .C("gvectorAdd",
       as.double(w),
       as.integer(net_shape),
       as.integer(n_layers),
       as.double(Finc),
       as.integer(f_count_in),
       as.integer(f_max_R),
       as.double(Flast),
       as.integer(t_steps),
       as.double(t_eps),
       as.double(gamma),
       as.double(gammad),
       as.integer(debug),
       as.logical(copy_gamma));

    if (debug >= 1) {
        print("R has control.")
    }

    # Unpack some results
    Flast_d <- rst[[7]]
    gamma_d <- rst[[10]]
    gammad_d <- rst[[11]]
    Fout <- list()
    GAMMA <- list()
    GAMMAd <- list()
    counts <- c(0, cumsum(f_max[[n_layers]]))
    counter = 0
    for (n in 1:net_shape[n_layers]) {
        # Postprocess Flast from a flat double array to a list of doubles
        Fout[[n]] <- Flast_d[(counts[n]+1):counts[n+1]]
        # And gammas as well
        GAMMA[[n]] <- list()
        GAMMAd[[n]] <- list()
        for (fi in 1:f_max[[length(f_max)]][n]) {
            GAMMA[[n]][[fi]] <- list()
            GAMMAd[[n]][[fi]] <- list()
            for (l in 1:n_layers) {
                GAMMA[[n]][[fi]][[l]] <- rep(NA, net_shape[l])
                GAMMAd[[n]][[fi]][[l]] <- rep(NA, net_shape[l])
                for (h in 1:net_shape[l]) {
                    counter <- counter + 1
                    GAMMA[[n]][[fi]][[l]][h] <- gamma_d[counter]
                    GAMMAd[[n]][[fi]][[l]][h] <- gammad_d[counter]
                }
            }
        }
    }
    # Purge firing events which were not achieved (denoted by -1).
    for (n in 1:net_shape[n_layers]) {
        if (sum(Fout[[n]]==-1) > 0) {
            Fout[[n]] <- Fout[[n]][which(Fout[[n]]!=-1)]
            GAMMA[[n]] <- GAMMA[[n]][which(Fout[[n]]!=-1)]
            GAMMAd[[n]] <- GAMMAd[[n]][which(Fout[[n]]!=-1)]
        }
    }
    if (copy_gamma) {
        return(list(Fout = Fout, GAMMA = GAMMA, GAMMAd = GAMMAd))
    } else {
        return(list(Fout = Fout))
    }
}
