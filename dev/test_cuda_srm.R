#!/usr/bin/Rscript
#  dev/read_in.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 10.29.2018

## See if we can transfer data to a cuda program.

## The Spike Response Model
I_0 <- 1 #Stimulation
tau <- 1
n_neurons <- 3
ipostkern <- function(dt) as.numeric(dt>=0) * tau * (1 - exp(-dt/tau))# Integrated kernel
iprekern <- function(dt) as.numeric(dt>=0) * -v_thresh# Integrated kernel


t_eps <- 0.1
t_end <- 3.5
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

v_thresh <- 1.5
v_reset <- 0

n_in <- 2
n_out <- 1
n_h <- c(1)
layers <- 2 + length(n_h)
#Ws <- list(matrix(c(3.5), ncol = 1), matrix(c(3), ncol = 1))

Fin <- list(seq(0,1, by = 1), seq(0,1, by = 1))

# Generate random wieghts
set.seed(123)
sizes <- c(n_in, n_h, n_out)
Ws <- lapply(1:(length(sizes)-1), function(i) 
             matrix(rnorm(sizes[i]*sizes[i+1], 3), nrow = sizes[i], ncol = sizes[i+1]))

# Calculate an upper bound on how many times a neuron will fire.
predict_fire_counts <- function(Ws, Fin) {
    counts <- list()
    last_count <- sapply(Fin, length)
    counts[[1]] <- last_count
    for (l in 1:length(Ws)) {
        counts[[l+1]] <- floor(tau / v_thresh * t(Ws[[l]]) %*% counts[[l]])
    }
    return(counts)
}

n <- 10
a <- rnorm(n)
b <- rnorm(n)

dyn.load('dev/read_in.so')

vector_add <- function(a, b) {
    n <- length(a)
    c <- rep(0, n)
    rst <- .C("gvectorAdd",
       as.double(a),
       as.double(b),
       as.double(c),
       as.integer(n))
    return(rst[[3]])
}
d <- vector_add(a, b)




n_in <- 2
n_out <- 1
n_h <- c(2,2)
net_shape <- c(n_in, n_h, n_out)
#Ws <- list(matrix(c(3.5), ncol = 1), matrix(c(3), ncol = 1))


# Generate random wieghts
set.seed(123)
Fin <- lapply(1:n_in, function(a) runif(sample(1:10, 1)))
sizes <- c(n_in, n_h, n_out)
Ws <- lapply(1:(length(sizes)-1), function(i) 
             matrix(rnorm(sizes[i]*sizes[i+1], 3), nrow = sizes[i], ncol = sizes[i+1]))

dyn.load('dev/read_in.so')

# Calculate an upper bound on how many times a neuron will fire.
predict_fire_counts <- function(Ws, Fin) {
    counts <- list()
    last_count <- sapply(Fin, length)
    counts[[1]] <- last_count
    for (l in 1:length(Ws)) {
        counts[[l+1]] <- as.numeric(floor(tau / v_thresh * t(Ws[[l]]) %*% counts[[l]]))
    }
    return(counts)
}

v_thresh <- 1.5
tau <- 1
f_max <- predict_fire_counts(Ws, Fin)

do_Ws <- function(Ws, net_shape, Fin) {
    n <- length(net_shape)
    w <- unlist(Ws)
    c <- rep(0, n)
    Finc <- unlist(Fin)
    f_count_in <- sapply(Fin, length)
    f_max_R <- unlist(f_max)
    rst <- .C("gvectorAdd",
       as.double(w),
       as.integer(net_shape),
       as.integer(n),
       as.double(Finc),
       as.integer(f_count_in),
       as.integer(f_max_R));
}

do_Ws(Ws, net_shape, Fin)


dyn.load("dev/read_in.so")
.C("cool")
