#!/usr/bin/Rscript
#  dev/test_inverse.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.03.2018

source('R/srm.R')

## The Spike Response Model
ipostkern <- function(dt) as.numeric(dt>=0) * tau * (1 - exp(-dt/tau))# Integrated kernel
postkern <- function(dt) as.numeric(dt>=0) * exp(-dt/tau)# Simply the kernel itself
dpostkernd <- function(dt) as.numeric(dt>=0) * (-1)/tau * exp(-dt/tau)# Derivative of the kernel 
iprekern <- function(dt) as.numeric(dt>=0) * -v_thresh# Integrated kernel

t_eps <- 0.01
t_end <- 5
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

v_thresh <- 1.5

n_in <- 1000
n_out <- 2
n_h <- c(10, 10)
net_shape <- c(n_in, n_h, n_out)
n_layers <- 2 + length(n_h)

# Seed all random things:
set.seed(12345)
# Generate random wieghts
sizes <- c(n_in, n_h, n_out)
Ws <- init_weights(net_shape, a = 3, inter = 0.3)
# Fire with uniform probabily on the interval, with a uniform number of firing events with max_input_fire max and 0 min
max_input_fire <- 10
Fin <- lapply(1:n_in, function(n) runif(sample(max_input_fire, 1), 0, t_end))
f_max <- predict_fire_counts(Ws, Fin)
f_max[[n_layers]]

