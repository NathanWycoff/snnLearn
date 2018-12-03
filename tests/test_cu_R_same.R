#!/usr/bin/Rscript
#  dev/srm_cu_R_same.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.25.2018

source('R/srm.R')

## The Spike Response Model
I_0 <- 1 #Stimulation
tau <- 1
n_neurons <- 3

t_eps <- 0.001
t_end <- 3.5
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

v_thresh <- 1.5
v_reset <- 0

n_in <- 2
n_out <- 2
n_h <- c(10, 5)
net_shape <- c(n_in, n_h, n_out)
layers <- 2 + length(n_h)
#Ws <- list(matrix(c(3.5), ncol = 1), matrix(c(3), ncol = 1))

# Fire with uniform probabily on the interval, with a uniform number of firing events with max_input_fire max and 0 min
max_input_fire <- 10
Fin <- lapply(1:n_in, function(n) runif(sample(max_input_fire, 1), 0, t_end))

# Generate random wieghts
set.seed(123)
sizes <- c(n_in, n_h, n_out)
Ws <- lapply(1:(length(sizes)-1), function(i) 
             matrix(rnorm(sizes[i]*sizes[i+1], 3), nrow = sizes[i], ncol = sizes[i+1]))

dyn.load('src/srm0.so')

system.time(a <- srm0_cu(Ws, net_shape, Fin, t_steps, t_eps, debug = 1))
#a <- srm0_R(Ws, Fin, t_steps)
#b <- srm0_R(Ws, net_shape, Fin, t_steps, t_eps)[[1]][[layers]]
system.time(b <- srm0_R(Ws, net_shape, Fin, t_steps, t_eps))
