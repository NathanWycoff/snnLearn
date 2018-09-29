#!/usr/bin/Rscript
#  R/split_network.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.17.2018

n_in <- 20
n_out <- 2
n_h <- c(10, 10, 10, 10)
sizes <- c(n_in, n_h, n_out)

n_proc <- 2

#' Prepare Feedforward Network for Parallel Processing
#'
#' Splits so that cores / layer is about constant, frontloading earlier layers with excess cores.
#'
#' @param Ws A list of matrices giving connection weights.
#' @param Fin The input spikes
#' @param sizes Vector of size of each layer.
#' @param n_proc The number of processors which will be handling the job.
split_snn <- function(Ws, Fin, sizes, n_proc) {

    # Split layers accross processors.
    L <- length(sizes) - 1
    procs <- lapply(1:n_proc, function(p) list(layers = c(), neurons = c()))
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
        print(size)
        print(n_proc_l)
        if (n_proc_l < size) {
            ends <- sapply(1:n_proc_l, function(i) ceiling(size / n_proc_l * i))
            ends <- c(0, ends)
            it <- 1
            for (p in pil[[l]]) {
                procs[[p]]$neurons[[paste('l', l, sep = '')]] <- (ends[it]+1):ends[it+1]
                it <- it + 1
            }
        } 
    }
}
