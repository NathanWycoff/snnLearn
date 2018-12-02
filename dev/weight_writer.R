#!/usr/bin/Rscript
#  dev/weight_writer.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.26.2018
source('R/srm.R')

## Write weights, net_shape, and input to a file.
I_0 <- 1 #Stimulation
tau <- 1
n_neurons <- 3
ipostkern <- function(dt) as.numeric(dt>=0) * tau * (1 - exp(-dt/tau))# Integrated kernel
iprekern <- function(dt) as.numeric(dt>=0) * -v_thresh# Integrated kernel

t_eps <- 0.001
t_end <- 3.5
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

v_thresh <- 1.5
v_reset <- 0

n_in <- 10
n_out <- 10
n_h <- c(10, 30, 40)
net_shape <- c(n_in, n_h, n_out)
layers <- 2 + length(n_h)
#Ws <- list(matrix(c(3.5), ncol = 1), matrix(c(3), ncol = 1))

Fin <- lapply(1:n_in, function(l) runif(sample(2:20, 1), 0, t_end))

# Generate random wieghts
set.seed(123)
sizes <- c(n_in, n_h, n_out)
Ws <- lapply(1:(length(sizes)-1), function(i) 
             matrix(rnorm(sizes[i]*sizes[i+1], 0.14, 0.1), nrow = sizes[i], ncol = sizes[i+1]))

f_max <- predict_fire_counts(Ws, Fin)

shape_str <- paste(net_shape, collapse = ",")
# Stores weights with one line per matrix in col major order, spaces between numbers and ';' denotes end of row.
weight_str <- paste(lapply(Ws, function(W) paste(lapply(1:ncol(W), function(col) paste(W[,col], collapse = ' ')), collapse = ';')), collapse = '\n')
weight_file_dat <- paste(shape_str, '\n', weight_str, sep = '')
write(weight_file_dat, file = "weights_file.dat")

f_max_file <- paste(lapply(f_max, function(fm) paste(fm, collapse = ' ')), collapse = ';')
write(f_max_file, file = 'f_max_file.dat')

Fin_file <- paste(lapply(Fin, function(i) paste(i, collapse = ' ')), collapse = '\n')
write(Fin_file, file = 'input_ap.dat')
