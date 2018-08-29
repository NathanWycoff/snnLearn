#!/usr/bin/Rscript
#  R/verify_analytic.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 08.27.2018

# Match our analytic solution

leak <- 0
tau <- 1
kern <- function(dt) exp(-dt/tau)
kernd <- function(dt) -1/tau * exp(-dt/tau)

t_eps <- 0.01
t_end <- 20
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

v_thresh <- 1.5
v_reset <- 0

w_init <- 1

go <- function(w) {
    v1s <- rep(NA, t_steps+1)
    v1s[1] <- v_reset
    v2s <- rep(NA, t_steps+1)
    v2s[1] <- v_reset
    fires1 <- c()
    fires2 <- c()
    for (ti in 1:length(ts)) {
        t <- ts[ti]

        input1 <- 1.0
        input2 <- w * sum(as.numeric(sapply(fires1, function(tt) kern(t-tt))))

        dv1dt <- - leak*v1s[ti] + input1
        dv2dt <- - leak*v2s[ti] + input2
        v1s[ti+1] <- v1s[ti] + t_eps * dv1dt
        v2s[ti+1] <- v2s[ti] + t_eps * dv2dt

        if (v1s[ti+1] > v_thresh) {
            v1s[ti+1] <- 0
            fires1 <- c(fires1, t)
        }
        if (v2s[ti+1] > v_thresh) {
            v2s[ti+1] <- 0
            fires2 <- c(fires2, t)
        }
    }
    par(mfrow=c(1,2))
    plot(ts, v1s[2:(t_steps+1)])
    plot(ts, v2s[2:(t_steps+1)])
    abline(v = fires2[1], col = 'red')
    return(list(fires1, fires2))
}

go(1)
