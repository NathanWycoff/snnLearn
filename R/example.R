#!/usr/bin/Rscript
#  R/c_layer_lif.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 08.30.2018

require(snnLearn)

## A LIF with layer structure
learn_rate <- 0.1

t_eps <- 0.01
t_end <- 10
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

n_in <- 2
n_out <- 1
n_h <- c(1)
layers <- 2 + length(n_h)
#Ws <- list(matrix(c(3.5), ncol = 1), matrix(c(3), ncol = 1))

Fin <- list(seq(0,10, by = 1), seq(0,10, by = 1))

# Generate random wieghts
set.seed(123)
sizes <- c(n_in, n_h, n_out)
Ws <- lapply(1:(length(sizes)-1), function(i) 
             matrix(rnorm(sizes[i]*sizes[i+1], 3), nrow = sizes[i], ncol = sizes[i+1]))

l_h <- length(n_h)
a <- goc(n_in, l_h, n_h, n_out, Ws, Fin, t_eps, t_steps) 
