#!/usr/bin/Rscript
#  examples/compare_speed.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 10.02.2018

## Compare the speed of existing implementations
require(snnLearn)
require(ggplot2)

learn_rate <- 0.1
I_0 <- 1 #Stimulation
leak <- 0.5
tau <- 1
n_neurons <- 3

t_eps <- 0.1
t_end <- 3.5
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

v_thresh <- 1.5
v_reset <- 0

hs <- seq(2, 200, by = 50)

timesa <- c()
timesb <- c()
for (h in hs) {
    n_in <- 2
    n_out <- 1
    n_h <- c(h, h, h, h, h)
    layers <- 2 + length(n_h)
    #Ws <- list(matrix(c(3.5), ncol = 1), matrix(c(3), ncol = 1))

    Fin <- list(seq(0,10, by = 1), seq(0,10, by = 1))

    # Generate random wieghts
    set.seed(123)
    sizes <- c(n_in, n_h, n_out)
    Ws <- lapply(1:(length(sizes)-1), function(i) 
                 matrix(rnorm(sizes[i]*sizes[i+1], 3), nrow = sizes[i], ncol = sizes[i+1]))
    l_h <- length(n_h)

    tta <- system.time(a <- lif_forward_R(n_in, l_h, n_h, n_out, Ws, Fin, t_eps, t_steps))
    ttb <- system.time(b <- lif_forward_Cpp(n_in, l_h, n_h, n_out, Ws, Fin, t_eps, t_steps))

    timesa <- c(timesa, tta[3])
    timesb <- c(timesb, ttb[3])
}

plotdf <- data.frame(rep(hs, 2), c(timesa, timesb), group = rep(c('R', 'C'), c(length(hs), length(hs))))
colnames(plotdf) <- c('h', 'time', 'imp')

gg <- ggplot(data = plotdf, aes(x = h, y = time, group = imp, color = imp)) + geom_line() + geom_point() + 
    xlab("number of hidden units per layer") + 
    ylab( "time in seconds") + 
    ggtitle("problem difficulty versus execution time")
ggsave('images/speed_comp.pdf', gg)
