#!/usr/bin/Rscript
#  R/larger_probs.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.04.2018
## Measure arithmetic intensity

source('R/srm.R')

Ns <- c(1000, 2500, 5000, 7500, 10000)
times <- c()

load("data/large_timing.RData")

for (NNN in Ns) {
    tau <- 1
    t_eps <- 0.01
    t_end <- 10
    ts <- seq(0, t_end, by = t_eps)
    t_steps <- length(ts)

    v_thresh <- 1.5
    v_reset <- 0

    n_in <- 10
    n_out <- 2
    n_h <- c(NNN, NNN)
    net_shape <- c(n_in, n_h, n_out)
    layers <- 2 + length(n_h)
    #Ws <- list(matrix(c(3.5), ncol = 1), matrix(c(3), ncol = 1))

    # Fire with uniform probabily on the interval, with a uniform number of firing events with max_input_fire max and 0 min
    set.seed(123)
    max_input_fire <- 10
    t_end_inputs <- 1
    Fin <- lapply(1:n_in, function(n) runif(sample(max_input_fire, 1), 0, t_end_inputs))
    #Fin <- lapply(1:n_in, function(n) rep(0, max_input_fire))
    # Generate random wieghts
    sizes <- c(n_in, n_h, n_out)
    #Ws <- lapply(1:(length(sizes)-1), function(i) 
    #             matrix(rgamma(net_shape[i]*net_shape[i+1],1,1), nrow = sizes[i], ncol = sizes[i+1]))
    Ws <- lapply(1:(length(sizes)-1), function(i) 
                 matrix(1, nrow = sizes[i], ncol = sizes[i+1]))
    #Ws <- lapply(Ws, function(W) W / svd(W)$d[1])
    Ws <- scale_Ws(Ws, Fin)
    Ws <- lapply(Ws, function(W) W * 4)
    f_max <- predict_fire_counts(Ws, Fin)
    f_max[[layers]]

    dyn.load('src/srm0.so')
    tt <- system.time(ret <- srm0_cu(Ws, net_shape, Fin, t_steps, t_eps))
    cat("Firing Counts:\n")
    print(sapply(ret$Fout, length))

    time <- tt[3]
    times <- c(times, time)

    save(Ns, times, file = "data/large_timing2.RData")
}
