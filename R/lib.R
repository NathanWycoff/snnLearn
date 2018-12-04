#!/usr/bin/Rscript
#  R/lib.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.25.2018

#' Calculate an upper bound on how many times a neuron will fire.
#'
#' Note: Returns 1 instead of 0 if 0 is calculated
#'
#' @param Ws The weight matrix list.
#' @param Fin The input firing times.
#' @param v_thresh The firing threshold
#' @param tau The time constant for the kernel.
predict_fire_counts <- function(Ws, Fin, v_thresh = 1.5, tau = 1) {
    counts <- list()
    last_count <- sapply(Fin, length)
    counts[[1]] <- last_count
    for (l in 1:length(Ws)) {
        counts[[l+1]] <- pmax(1, as.numeric(floor(tau / v_thresh * t(Ws[[l]]) %*% counts[[l]])))
    }
    return(counts)
}

#' Initialize the Weight Matrix
#' As in https://arxiv.org/pdf/1608.08782.pdf
#' @param net_shape
init_weights <- function(net_shape, a = 3, inter = 0.1) {
    layers <- length(net_shape)
    Ws <- list()
    for (l in 1:(layers-1)) {
        b <- sqrt(3 / net_shape[l])
        Ws[[l]] <- inter + matrix(runif(net_shape[l]*net_shape[l+1], -b, b), nrow = net_shape[l], ncol = net_shape[l+1])
    }
    return(Ws)
}

#' Scale weights so that there is a constant amoung of firing at every level of the network.
scale_Ws <- function(Ws, Fin) {
    vec_norm <- function(x) sqrt(sum(x^2))
    Wsn <- Ws
    gamma <- sapply(Fin, length)
    for (l in 1:length(Ws)) {
        Wc <- (tau / v_thresh * t(Ws[[l]]) %*% gamma)
        Wsn[[l]] <- Wsn[[l]] * vec_norm(gamma) / vec_norm(Wc)
        gamma <- Wc
    }
    return(Wsn)
}
