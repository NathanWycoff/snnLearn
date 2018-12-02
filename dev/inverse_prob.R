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

n_in <- 3
n_out <- 2
n_h <- c(2)
net_shape <- c(n_in, n_h, n_out)
n_layers <- 2 + length(n_h)
#Ws <- list(matrix(c(3.5), ncol = 1), matrix(c(3), ncol = 1))

# Seed all random things:
set.seed(123)

# Generate random wieghts
sizes <- c(n_in, n_h, n_out)
Ws <- lapply(1:(length(sizes)-1), function(i) 
             matrix(rnorm(sizes[i]*sizes[i+1], 3), nrow = sizes[i], ncol = sizes[i+1]))

# Fire with uniform probabily on the interval, with a uniform number of firing events with max_input_fire max and 0 min
max_input_fire <- 10
Fin <- lapply(1:n_in, function(n) runif(sample(max_input_fire, 1), 0, t_end))


ret <- srm0_R(Ws, net_shape, Fin, t_steps, t_eps)

#TODO: generalize t_desired to multiple per point
t_desired <- c(1.69, 4.20)

iters <- 1000
learn_rate <- 0.01
last_Ws <- Ws
grad_norm <- Inf

for (iter in 1:iters) {
    #Assumes at least 1 hidden layer
    #ret <- gon(Ws, Fin)
    ret <- srm0_R(Ws, net_shape, Fin, t_steps, t_eps)
    Fout <- ret$Fout
    GAMMA <- ret$GAMMA
    GAMMAd <- ret$GAMMAd

    #TOOD: More than 1 target neuron
    #target_neuron <- 1
    #ta <- Fout[[target_neuron]][1]
    #tai <- which(abs(ts-ta) < t_eps/2)
    t_actuals <- sapply(Fout, function(fs) fs[1])
    print(paste("Iter:", iter, "has firing times", paste(t_actuals, collapse = ", "), "with norm", grad_norm))

    #TODO: Modify next line for firings per output
    if (any(sapply(Fout, length) == 0)) {
        print("No firing events detected, halving learning rate and disregarding last iter...")
        learn_rate <- learn_rate / 2
        print(paste("New learn rate:", learn_rate))
        Ws <- last_Ws
        next
    }

    # Output delta
    d_out <- rep(NA, n_out)
    for (n in 1:n_out) {
        #TODO: one non assumption: modify td & ta
        #d_out[n] <- -(ta - td) / t(Ws[[length(n_h)+1]]) %*% GAMMAd[[length(n_h)+1]][,tai]

        # TODO: The other 1 says "only look at the first firing event"
        d_out[n] <- -(t_actuals[n] - td[n]) / t(Ws[[length(n_h)+1]][,n]) %*% GAMMAd[[n]][[1]][[length(n_h)+1]]
    }

    # Hidden Delta
    d_h <- lapply(n_h, function(h) matrix(NA, nrow = h, ncol = n_out))
    for (l in length(n_h):1) {
        for (neur in 1:n_h[l]) {
            #TODO: d_out not good for more than 1 hidden layer
            #d_h[[l]][neur] <-  d_out * (GAMMAd[[l+1]][neur,tai] * sum(Ws[[l+1]][neur,]))  /
            #    t(Ws[[l]][,neur]) %*% GAMMAd[[l]][,tai]
            # TODO: The other 1 says "only look at the first firing event"
            d_h[[l]][neur,] <-  d_out * (GAMMAd[[1]][[1]][[l+1]][neur] * sum(Ws[[l+1]][neur,]))  /
                t(Ws[[l]][,neur]) %*% GAMMAd[[1]][[1]][[l]]
        }
    }
    delta <- d_h
    delta[[length(delta)+1]] <- matrix(d_out, nrow = 1)

    # Calculate weight updates, and apply them
    last_Ws <- Ws
    grad_norm <- 0
    for (wi in 1:length(Ws)) {
        #Wd <- -t(delta[[wi]]) %x% GAMMA[[wi]][,tai]
        # TODO: the 1 in the next line means first firing event, and should be changed in the future.
        # Each columns has its own gradient for output layers, summed for the others
        # since we're averaging gradients for hidden layers, we should do the same for the output weights.
        if (wi == length(Ws)) {
            for (on in 1:n_out) {
                Wd <- -t(delta[[wi]][,on]) %x% GAMMA[[on]][[1]][[wi]] / n_out
                grad_norm <- grad_norm + sum(Wd^2)
                Ws[[wi]][,on] <- Ws[[wi]][,on] - learn_rate * Wd
            }
        } else {
            Wd <- 0
            for (on in 1:n_out) {
                Wd <- Wd + -t(delta[[wi]][,on]) %x% GAMMA[[on]][[1]][[wi]]
            }
            # Scale by number of output neurons to keep gradient size similar
            Wd <- Wd / n_out
            grad_norm <- grad_norm + sum(Wd^2)
            Ws[[wi]] <- Ws[[wi]] - learn_rate * Wd
        }
    }
} 
