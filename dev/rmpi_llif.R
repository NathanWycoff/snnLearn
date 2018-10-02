#!/usr/bin/Rscript
#  dev/rmpi_llif.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.29.2018

require('Rmpi')

mpi.spawn.Rslaves(nslaves = 4)

# Rank 0 partitions the work and sends it out to everyone.
output <- mpi.remote.exec({
    source('dev/split_network.R')
    source('dev/mpi_integrate_lif.R')

    # Set up network params
    #TODO: everybody exectutes this, and we probably don't want that
    set.seed(123)
    t_eps <- 0.1
    t_end <- 3.5
    ts <- seq(0, t_end, by = t_eps)
    t_steps <- length(ts)
    n_in <- 3
    n_out <- 2
    n_h <- c(2, 3)
    v_thresh <- 1.5
    leak <- 0.5
    sizes <- c(n_in, n_h, n_out)
    Ws <- lapply(1:(length(sizes)-1), function(i)
                matrix(rnorm(sizes[i]*sizes[i+1], 3), nrow = sizes[i], ncol = sizes[i+1]))
    Fin <- lapply(1:n_in, function(i) seq(0,10, by = 1))

    proc <- mpifun_split_snn(Ws, Fin, sizes)

    proc <- mpifun_setup_int(proc, sizes, t_steps)

    # Integrate ODE system using Forward Euler
    t <- 0
    for (ti in 1:length(ts)) {

        # Do 1 time step
        ret <- mpifun_odestep(proc, t, leak, t_eps, v_thresh)
        proc <- ret$proc
        post_Fcal <- ret$post_Fcal
        t <- t + t_eps

        # Tell downstream neurons about firing events
        #TODO: This can be done as a pure mpi call
        for (psn in proc$postsyn) {
            mpi.isend.Robj(post_Fcal, psn, tag = 42069)
        }

        # Listen to upstream neurons about their firing events.
        pre_Fcals <- list()
        for (psn in proc$presyn) {
            pre_Fcals[[length(pre_Fcals)+1]] <- mpi.recv.Robj(psn, tag = 42069)
        }

        # Store this info in Fcal
        for (report in pre_Fcals) {
            if (length(report) > 0) {
                for (i in 1:(length(report)/2)) {
                    layer <- which(proc$presyn==report[2*(i-1) + 1])
                    neuron <- report[2*(i-1) + 2]
                    proc$Fcal[[layer]][[neuron]] <- c(proc$Fcal[[layer]][[neuron]], t + t_eps)
                }
            } 
        }
    }
    proc
})

mpi.quit()
