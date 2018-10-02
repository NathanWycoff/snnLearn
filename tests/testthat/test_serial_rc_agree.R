context("R and C output agree")
library(snnLearn)

test_that("R and C output agree", {

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

    a <- lif_forward_R(n_in, l_h, n_h, n_out, Ws, Fin, t_eps, t_steps) 
    b <- lif_forward_Cpp(n_in, l_h, n_h, n_out, Ws, Fin, t_eps, t_steps) 
    #a <- lif_forward_R(n_in, l_h, n_h, n_out, Ws, Fin, t_eps, t_steps)$Fcal 
    #b <- goc(n_in, l_h, n_h, n_out, Ws, Fin, t_eps, t_steps) 
    
    expect_equal(length(a), length(b))
    for (i in 1:length(a)) {
        la <- a[[i]]
        lb <- b[[i]]
        expect_equal(length(la), length(lb))
        for (j in 1:length(la)) {
            expect_equivalent(la[[j]], lb[[j]])
        }
    }
})
