source('R/srm.R')

## The Spike Response Model
I_0 <- 1 #Stimulation
tau <- 1
n_neurons <- 3
ipostkern <- function(dt) as.numeric(dt>=0) * tau * (1 - exp(-dt/tau))# Integrated kernel
postkern <- function(dt) as.numeric(dt>=0) * exp(-dt/tau)# Simply the kernel itself
dpostkernd <- function(dt) as.numeric(dt>=0) * (-1)/tau * exp(-dt/tau)# Derivative of the kernel 
iprekern <- function(dt) as.numeric(dt>=0) * -v_thresh# Integrated kernel

t_eps <- 0.01
t_end <- 5
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

v_thresh <- 1.5
v_reset <- 0

n_in <- 5
n_out <- 2
n_h <- c(10, 10)
net_shape <- c(n_in, n_h, n_out)
n_layers <- 2 + length(n_h)
#Ws <- list(matrix(c(3.5), ncol = 1), matrix(c(3), ncol = 1))

# Seed all random things:
set.seed(1234)

# Generate random wieghts
sizes <- c(n_in, n_h, n_out)
Ws <- lapply(1:(length(sizes)-1), function(i) 
             matrix(rnorm(sizes[i]*sizes[i+1], 1.5), nrow = sizes[i], ncol = sizes[i+1]))

# Fire with uniform probabily on the interval, with a uniform number of firing events with max_input_fire max and 0 min
max_input_fire <- 10
Fin <- lapply(1:n_in, function(n) runif(sample(max_input_fire, 1), 0, t_end))

dyn.load('src/srm0.so')
system.time(ret <- srm0_R(Ws, net_shape, Fin, t_steps, t_eps))
system.time(ret <- srm0_cu(Ws, net_shape, Fin, t_steps, t_eps))

#TODO: generalize t_desired to multiple per point
#TODO: can't yet handle the case where there are fewer observed firings than desired at the beginning
#t_desired <- list(c(1,2,3), c(1.5, 3), c(4.2)) #A list of numeric vectors as long as the output layer, each giving a desired firing time.
t_desired <- list(c(4.5), c(0.5)) #A list of numeric vectors as long as the output layer, each giving a desired firing time.

t_ns <- sapply(t_desired, length)

iters <- 1000
learn_rate <- 10
last_Ws <- Ws
grad_norm <- Inf

for (iter in 1:iters) {
    #Assumes at least 1 hidden layer
    #ret <- srm0_R(Ws, net_shape, Fin, t_steps, t_eps)
    ret <- srm0_cu(Ws, net_shape, Fin, t_steps, t_eps)

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

    print(paste("Iter:", iter, "has firing times", paste(t_actuals, collapse = ", "), "with gradient norm", grad_norm, "|| Error:", err))

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
    last_Ws <- Ws
    grad_norm <- 0
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
                Wd <- Wd / sum(t_ns)
                grad[[length(grad)+1]] <- Wd
                grad_norm <- grad_norm + sum(abs(Wd))
                Ws[[wi]][,on] <- Ws[[wi]][,on] - learn_rate * Wd
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
            Wd <- Wd / sum(t_ns)
            grad[[length(grad)+1]] <- Wd
            grad_norm <- grad_norm + sum(abs(Wd))
            Ws[[wi]] <- Ws[[wi]] - learn_rate * Wd
        }
    }
    #print(grad)
} 
