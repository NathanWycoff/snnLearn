#!/usr/bin/Rscript
#  R/split_network.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.17.2018

require('Rmpi')

#' Prepare Feedforward Network for Parallel Processing
#'
#' Splits so that cores / layer is about constant, frontloading earlier layers with excess cores.
#'
#' @param Ws A list of matrices giving connection weights.
#' @param Fin The input spikes, a list of numeric vectors giving firing times.
#' @param sizes Vector of size of each layer.
#' @param n_proc The number of processors which will be handling the job.
#' @return A list of lists, of length n_proc, each sublist has components layers, giving which layers blong to a processors, neurons, a list of integer vectors, giving which neurons belong within that layer (of length equal to length(sizes)-1, so most will be empty for any particular layer), postsyn, giving which layers are postsynaptic to it, and Ws, a list of presynpatic weights for the layers allocated to the processor.
split_snn <- function(Ws, Fin, sizes, n_proc) {

    # Split layers accross processors.
    L <- length(sizes) - 1
    procs <- lapply(1:n_proc, function(p) list(layers = c(), neurons = list()))
    it <- 0
    pil <- lapply(1:(L), function(s) c())# Gives processor assigned to each layer
    to <- max(L, n_proc)
    for (it in 0:(to-1)) {
        selp <- it %% n_proc + 1
        sell <- it %% L + 1
        procs[[selp]]$layers <- c(procs[[selp]]$layers, sell)
        pil[[sell]] <- c(pil[[sell]], selp)
    }

    # Split neurons within layers accross processors
    for (l in 1:(length(sizes)-1)) {
        size <- sizes[l+1]
        n_proc_l <- length(pil[[l]])
        if (n_proc_l < size) {
            ends <- sapply(1:n_proc_l, function(i) ceiling(size / n_proc_l * i))
            ends <- c(0, ends)
            it <- 1
            for (p in pil[[l]]) {
                ind_of_l <- which(procs[[p]]$layers==l)
                procs[[p]]$neurons[[ind_of_l]] <- (ends[it]+1):ends[it+1]
                it <- it + 1
            }
        } else {
            neur <- 1
            for (p in pil[[l]][1:size]) {
                ind_of_l <- which(procs[[p]]$layers==l)
                procs[[p]]$neurons[[ind_of_l]] <- neur
                neur <- neur + 1
            }
        }
    }

    # Give each processor the weights it needs from presynaptic layers
    for (proc in 1:n_proc) {
        procs[[proc]]$postsyn <- Filter(function(x) x < length(sizes), 
                                        procs[[proc]]$layers + 1)
        procs[[proc]]$Ws <- lapply(1:length(procs[[proc]]$layers), 
                                   function(li) Ws[[procs[[proc]]$layers]][,procs[[proc]]$neurons[[li]]])
    }

    # Set up presynpatic firing times
    for (proc in 1:n_proc) {
        presyn <- procs[[proc]]$layers - 1
        procs[[proc]]$Fcal <- lapply(presyn, function(l) {
                                         if (l > 0) {
                                             lapply(1:sizes[l+1], function(i) list())
                                         } else {
                                             Fin
                                         }
                                        })
    }

    return(procs)
}

## Split up the network
mpifun_split_snn <- function(Ws, Fin, sizes) {
    if (mpi.comm.rank() == 1) {
        source('dev/split_network.R')
        procs <- split_snn(Ws, Fin, sizes, n_proc = mpi.comm.size()-1)

        if (mpi.comm.size() > 2) {
            for (r in 2:(mpi.comm.size()-1)) {
                mpi.send.Robj(procs[[r-1]], r, 42069)
            }
        }
        proc <- procs[[1]]
        print(proc)
    } else {
        proc <- mpi.recv.Robj(1, 42069)
    }

    return(proc)
}
