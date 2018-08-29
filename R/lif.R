#!/usr/bin/Rscript
#  R/lif.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 08.26.2018

## A simple LIF model.
learn_rate <- 0.1
I_0 <- 1 #Stimulation
leak <- 0.5
tau <- 1
n_neurons <- 3
kern <- function(dt) exp(-dt/tau)
kernd <- function(dt) -1/tau * exp(-dt/tau)

t_eps <- 0.01
t_end <- 10
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

v_thresh <- 1.5
v_reset <- 0

W <- matrix(c(0,0,0,1,3,0,0,0,0,3,0,0), ncol = n_neurons)

go <- function(W) {
    V = matrix(NA, nrow = n_neurons, ncol = t_steps+1)
    ALPHA = matrix(NA, nrow = n_neurons, ncol = t_steps)
    ALPHAd = matrix(NA, nrow = n_neurons, ncol = t_steps)
    V[,1] <- rep(0, n_neurons)
    Fcal <- lapply(1:n_neurons, function(i) c())

    for (ti in 1:length(ts)) {
        t <- ts[ti]

        # Calculate Post-synaptic Potential 
        ALPHA[,ti] <- sapply(Fcal, function(Fc) 
                   sum(as.numeric(sapply(Fc, function(tf) kern(t - tf)))))
        ALPHAd[,ti] <- sapply(Fcal, function(Fc) 
                   sum(as.numeric(sapply(Fc, function(tf) kernd(t - tf)))))

        # Calculate input
        input <- t(W) %*% c(ALPHA[,ti], I_0)

        # Calculate derivative
        dvdt <- - leak * V[,ti] + input

        # Update
        V[,ti+1] <- V[,ti] + t_eps * dvdt

        # Check for action potentials
        for (neur in 1:n_neurons) {
            if (V[neur,ti+1] > v_thresh) {
                V[neur,ti+1] <- 0
                Fcal[[neur]] <- c(Fcal[[neur]], t + t_eps)
            }
        }
    }
    return(list(Fcal, ALPHA, ALPHAd))
}

go(W)[[1]]

td <- 4.5

iters <- 100
for (iter in 1:iters) {
    fires <- go(w1_i, w2_i)
    fires1 <- fires[[1]]
    fires2 <- fires[[2]]
    fires3 <- fires[[3]]
    ta <- fires3[1]
    print(ta)
    abline(v = td, col = 'blue')

    F1cal <- fires1[fires1<=ta]
    F2cal <- fires2[fires2<=ta]

    # Update weight to output layer
    kern_cont <- sum(as.numeric(sapply(F2cal, function(tt) kern(ta-tt))))
    kernd_cont <- sum(as.numeric(sapply(F2cal, function(tt) kernd(ta-tt))))
    dw2 <- kern_cont * (ta - td) / (w2_i * kernd_cont)
    w2_i <- w2_i - learn_rate * dw2

    delta2 <- (ta - td) / kernd_cont

    # Update weight to hidden layer
    #kern_cont <- sum(as.numeric(sapply(F1cal, function(tt) kern(ta-tt))))
    #dw1 <- kern_cont * delta
    #w1_i <- w1_i - learn_rate * dw1
    #TODO: not sure that ta is actually appropriate in next 2 lines
    kern_conth <- sum(as.numeric(sapply(F1cal, function(tt) kern(ta-tt))))
    kernd_conth <- sum(as.numeric(sapply(F2cal, function(tt) kernd(ta-tt))))
    delta1 <- (delta2 * w2_i * kernd_cont) / (w1_i * kernd_conth)
    dw1 <- kern_conth * delta1
    w1_i <- w1_i - learn_rate * dw1
}
