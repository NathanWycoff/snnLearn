#!/usr/bin/Rscript
#  R/lib.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.25.2018

#' Calculate an upper bound on how many times a neuron will fire.
#' @param Ws The weight matrix list.
#' @param Fin The input firing times.
#' @param v_thresh The firing threshold
#' @param tau The time constant for the kernel.
predict_fire_counts <- function(Ws, Fin, v_thresh = 1.5, tau = 1) {
    counts <- list()
    last_count <- sapply(Fin, length)
    counts[[1]] <- last_count
    for (l in 1:length(Ws)) {
        counts[[l+1]] <- as.numeric(floor(tau / v_thresh * t(Ws[[l]]) %*% counts[[l]]))
    }
    return(counts)
}

