#!/usr/bin/Rscript
#  R/lif_smol.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 08.27.2018

## A simple LIF model.
learn_rate <- 0.1
leak <- 0.5
tau <- 1
kern <- function(dt) exp(-dt/tau)
kernd <- function(dt) -1/tau * exp(-dt/tau)

t_eps <- 0.01
t_end <- 10
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

v_thresh <- 1.5
v_reset <- 0

w1_init <- 3.5
w2_init <- 3

go <- function(w1, w2) {
    v1s <- rep(NA, t_steps+1)
    v1s[1] <- v_reset
    v2s <- rep(NA, t_steps+1)
    v2s[1] <- v_reset
    v3s <- rep(NA, t_steps+1)
    v3s[1] <- v_reset
    fires1 <- c()
    fires2 <- c()
    fires3 <- c()
    for (ti in 1:length(ts)) {
        t <- ts[ti]

        input1 <- 1
        input2 <- w1 * sum(as.numeric(sapply(fires1, function(tt) kern(t-tt))))
        input3 <- w2 * sum(as.numeric(sapply(fires2, function(tt) kern(t-tt))))

        dv1dt <- - leak*v1s[ti] + input1
        dv2dt <- - leak*v2s[ti] + input2
        dv3dt <- - leak*v3s[ti] + input3
        v1s[ti+1] <- v1s[ti] + t_eps * dv1dt
        v2s[ti+1] <- v2s[ti] + t_eps * dv2dt
        v3s[ti+1] <- v3s[ti] + t_eps * dv3dt

        if (v1s[ti+1] > v_thresh) {
            v1s[ti+1] <- 0
            fires1 <- c(fires1, t + t_eps)
        }
        if (v2s[ti+1] > v_thresh) {
            v2s[ti+1] <- 0
            fires2 <- c(fires2, t + t_eps)
        }
        if (v3s[ti+1] > v_thresh) {
            v3s[ti+1] <- 0
            fires3 <- c(fires3, t + t_eps)
        }
    }
    par(mfrow=c(1,3))
    plot(ts, v1s[2:(t_steps+1)])
    plot(ts, v2s[2:(t_steps+1)])
    plot(ts, v3s[2:(t_steps+1)])
    abline(v = fires3[1], col = 'red')
    return(list(rbind(v1s, v2s, v3s), list(fires1, fires2, fires3)))
}

go(3.5,3)[[2]]

td <- 4.5
w1_i <- w1_init
w2_i <- w2_init

iters <- 100
for (iter in 1:iters) {
    fires <- go(w1_i, w2_i)[[2]]
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

    delta2 <- (ta - td) / (w2_i * kernd_cont)

    # Update weight to hidden layer
    #kern_cont <- sum(as.numeric(sapply(F1cal, function(tt) kern(ta-tt))))
    #dw1 <- kern_cont * delta
    #w1_i <- w1_i - learn_rate * dw1
    #TODO: not sure that ta is actually appropriate in next 2 lines
    kern_conth <- sum(as.numeric(sapply(F1cal, function(tt) kern(ta-tt))))
    kernd_conth <- sum(as.numeric(sapply(F1cal, function(tt) kernd(ta-tt))))
    delta1 <- (delta2 * w2_i * kernd_cont) / (w1_i * kernd_conth)
    dw1 <- kern_conth * delta1

    print(dw1)
    print(dw2)

    w1_i <- w1_i - learn_rate * dw1
    w2_i <- w2_i - learn_rate * dw2
}
