#!/usr/bin/Rscript
#  dev/xor.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.03.2018

## Solve the exclusive OR problem
source('R/srm.R')

t_eps <- 0.01
t_end <- 10
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

v_thresh <- 1.5

n_in <- 2
n_out <- 1
n_h <- c(4)
net_shape <- c(n_in, n_h, n_out)
n_layers <- 2 + length(n_h)

# Seed all random things:
set.seed(123)
# Generate random wieghts
sizes <- c(n_in, n_h, n_out)
Ws <- init_weights(net_shape, a = 3, inter = 1.5)

# Different Fin
F10 <- list(0, 3)
F01 <- list(3, 0)
F00 <- list(3, 3)
F11 <- list(0, 0)
Fins <- list(F10, F01, F00, F11)

targets <- list(list(5), list(5), list(9), list(9))

# Iterate between the targets
dyn.load('src/srm0.so')
f_max <- predict_fire_counts(Ws, Fins[[1]])
ret <- srm0_cu(Ws, net_shape, Fins[[3]], t_steps, t_eps, copy_gamma = TRUE)

iters <- 1000
learn_rate <- 0.01
last_Ws <- Ws
grad_norm <- 0
max_gnorm <- 20
accum <- 4 # Update params over these periods (minibatch size)
Wds <- lapply(1:length(Ws), function(l) matrix(0, nrow = net_shape[l], ncol = net_shape[l+1]))

for (iter in 1:iters) {

    # Pick a set of targets for this round
    ind <- (iter-1) %% 4 + 1
    #ind <- 1
    Fin <- Fins[[ind]]
    t_desired <- targets[[ind]]
    t_ns <- sapply(t_desired, length)

    #Assumes at least 1 hidden layer
    #ret <- srm0_R(Ws, net_shape, Fin, t_steps, t_eps, copy_gamma = TRUE)
    time <- Sys.time()
    ret <- srm0_cu(Ws, net_shape, Fin, t_steps, t_eps, copy_gamma = TRUE)

    Fout <- ret$Fout
    GAMMA <- ret$GAMMA
    GAMMAd <- ret$GAMMAd

    #TOOD: More than 1 target neuron
    #target_neuron <- 1
    #ta <- Fout[[target_neuron]][1]
    #tai <- which(abs(ts-ta) < t_eps/2)
    #t_actuals <- sapply(Fout, function(fs) fs[1])
    t_actuals <- lapply(1:n_out, function(n) Fout[[n]][1:t_ns[n]])
    err <- sum(sapply(1:n_out, function(n) sum((t_actuals[[n]] - t_desired[[n]])^2)))

    print(paste("Iter:", iter, "has firing times", paste(t_actuals, collapse = ", "), "with gradient norm", grad_norm, "|| Error:", err, "|| Time:", Sys.time()-time))
    print(paste("Index:", ind, "target:", t_desired))

    #TODO: Modify next line for firings per output
    if (any(sapply(Fout, length) - t_ns < 0)) {
        print("Insufficient firing events detected, halving learning rate and disregarding last iter...")
        learn_rate <- learn_rate / 2
        print(paste("New learn rate:", learn_rate))
        Ws <- last_Ws
        next
    }

    # Output delta
    #d_out <- rep(NA, n_out)
    d_out <- lapply(1:n_out, function(n) rep(NA, t_ns[n]))
    for (n in 1:n_out) {
        #TODO: one non assumption: modify td & ta
        #d_out[n] <- -(ta - td) / t(Ws[[length(n_h)+1]]) %*% GAMMAd[[length(n_h)+1]][,tai]

        # TODO: The other 1 says "only look at the first firing event"
        for (fi in 1:t_ns[n]) {
            dt <-  t_desired[[n]][fi] - t_actuals[[n]][fi]
            d_out[[n]][fi] <- dt / (t(Ws[[length(n_h)+1]][,n]) %*% GAMMAd[[n]][[fi]][[length(n_h)+1]])
        }
    }

    # Hidden Delta
    #d_h <- lapply(n_h, function(h) matrix(NA, nrow = h, ncol = n_out))
    # Still assumes 1 hidden layer
    d_h <- lapply(1:n_out, function(on) lapply(1:t_ns[on], function(f) lapply(n_h, function(h) rep(NA, h))))
    for (n in 1:n_out) {
        for (fi in 1:t_ns[n]) {
            for (l in length(n_h):1) {
                for (h in 1:net_shape[l+1]) {
                    #d_h[[n]][[fi]][[l]][h] <- d_out[[n]][fi] * (GAMMAd[[n]][[fi]][[l+1]] %*% Ws[[l+1]][h,])  /
                    #    t(Ws[[l]][,h]) %*% GAMMAd[[n]][[fi]][[l]]
                    # Note sure out of these two:
                    #num <- d_out[[n]][[fi]] * Ws[[l+1]][h,n]
                    if (l == length(n_h)) {
                        num <- d_out[[n]][[fi]] * Ws[[l+1]][h,n]
                    } else {
                        #num <- d_h[[n]][[fi]][[l+1]] %*% (Ws[[l]][h,] * GAMMAd[[n]][[fi]][[l+1]])
                        num <- (d_h[[n]][[fi]][[l+1]] %*% Ws[[l+1]][h,]) * GAMMAd[[n]][[fi]][[l+1]][h]
                    }
                    denom <- t(Ws[[l]][,h]) %*% GAMMAd[[n]][[fi]][[l]]
                    d_h[[n]][[fi]][[l]][h] <- num / denom
                }
            }
        }
    }

    # Calculate weight updates, and apply them
    grad <- list()
    for (wi in 1:length(Ws)) {
        #Wd <- -t(delta[[wi]]) %x% GAMMA[[wi]][,tai]

        # Each columns has its own gradient for output layers, summed for the others, so we need the input statement
        # since we're averaging gradients for hidden layers, we should do the same for the output weights.
        if (wi == length(Ws)) {
            next
            for (on in 1:n_out) {
                Wd <- 0
                for (fi in 1:t_ns[on]) {
                    #for (npre in 1:net_shape[wi]) {
                    #    for (npost in 1:net_shape[wi+1]) {
                    #        Wd[npre, npost] <- Wd[npre, npost] + delta[[n]][[fi]][[wi]][[npost]] * GAMMA[[n]][[fi]][[wi]][[npre]]
                    #    }
                    #}
                    # These are equivalent
                    Wd <- Wd + t(d_out[[on]][[fi]]) %x% GAMMA[[on]][[fi]][[wi]] 
                }
                Wd <- Wd / (accum*sum(t_ns))
                grad[[length(grad)+1]] <- Wd
                grad_norm <- grad_norm + sum(Wd^2)
                #Ws[[wi]][,on] <- Ws[[wi]][,on] - learn_rate * Wd
                Wds[[wi]] <- Wds[[wi]] + Wd
            }
        } else {
            Wd <- 0
            for (on in 1:n_out) {
                for (fi in 1:t_ns[on]) {
                    #for (npre in 1:net_shape[wi]) {
                    #    for (npost in 1:net_shape[wi+1]) {
                    #        Wd[npre, npost] <- delta[[n]][[fi]][[wi]][[npost]] * GAMMA[[n]][[fi]][[wi]][[npre]]
                    #    }
                    #}
                    # These are equivalent
                    Wd <- Wd + t(d_h[[on]][[fi]][[wi]]) %x% GAMMA[[on]][[fi]][[wi]]
                }
            }
            # Scale by number of output neurons to keep gradient size similar
            Wd <- Wd / (accum*sum(t_ns))
            grad[[length(grad)+1]] <- Wd
            grad_norm <- grad_norm + sum(Wd^2)
            #Ws[[wi]] <- Ws[[wi]] - learn_rate * Wd
            Wds[[wi]] <- Wds[[wi]] + Wd
        }
    }


    ## Update Weights
    if ((iter-1) %% accum == 0) {
        last_Ws <- Ws
        if (grad_norm > max_gnorm) {
            Wds <- lapply(Wds, function(Wd) Wd / sqrt(grad_norm) * max_gnorm)
            print("Notice: Norm Truncated.")
        }
        for (wi in 1:length(Ws)) {
            Ws[[wi]] <- Ws[[wi]] - learn_rate * Wds[[wi]]
        }
        Wds <- lapply(1:length(Ws), function(l) matrix(0, nrow = net_shape[l], ncol = net_shape[l+1]))
        grad_norm <- 0
    }
    #print(grad)
} 
