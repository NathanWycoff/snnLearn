#!/usr/bin/Rscript
#  dev/ai.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.04.2018

## Measure arithmetic intensity

source('R/srm.R')

Ts <- c(5, 10, 15)
Ns <- c(100, 500, 1000)

params <- c()
response <- c()
iters <- 2

for (TTT in Ts) {
    for (NNN in Ns) {
        mean_percents <- 0
        for (iter in 1:iters) {
            ## The Spike Response Model
            tau <- 1

            t_eps <- 0.01
            t_end <- TTT
            ts <- seq(0, t_end, by = t_eps)
            t_steps <- length(ts)

            v_thresh <- 1.5
            v_reset <- 0

            n_in <- 10
            n_out <- 2
            n_h <- c(NNN, NNN)
            net_shape <- c(n_in, n_h, n_out)
            layers <- 2 + length(n_h)
            #Ws <- list(matrix(c(3.5), ncol = 1), matrix(c(3), ncol = 1))

            # Fire with uniform probabily on the interval, with a uniform number of firing events with max_input_fire max and 0 min
            set.seed(123)
            max_input_fire <- 10
            t_end_inputs <- 1
            Fin <- lapply(1:n_in, function(n) runif(sample(max_input_fire, 1), 0, t_end_inputs))
            #Fin <- lapply(1:n_in, function(n) rep(0, max_input_fire))
            # Generate random wieghts
            sizes <- c(n_in, n_h, n_out)
            #Ws <- lapply(1:(length(sizes)-1), function(i) 
            #             matrix(rgamma(net_shape[i]*net_shape[i+1],1,1), nrow = sizes[i], ncol = sizes[i+1]))
            Ws <- lapply(1:(length(sizes)-1), function(i) 
                         matrix(1, nrow = sizes[i], ncol = sizes[i+1]))
            #Ws <- lapply(Ws, function(W) W / svd(W)$d[1])
            Ws <- scale_Ws(Ws, Fin)
            Ws <- lapply(Ws, function(W) W * 4)
            f_max <- predict_fire_counts(Ws, Fin)
            f_max[[layers]]

            ### This will be commented out some day
            #dyn.load('src/srm0.so')
            #system.time(ret <- srm0_cu(Ws, net_shape, Fin, t_steps, t_eps))
            #sapply(ret$Fout, length)

            ### Write weights to file
            shape_str <- paste(net_shape, collapse = ",")
            # Stores weights with one line per matrix in col major order, spaces between numbers and ';' denotes end of row.
            weight_str <- paste(lapply(Ws, function(W) paste(lapply(1:ncol(W), function(col) paste(W[,col], collapse = ' ')), collapse = ';')), collapse = '\n')
            weight_file_dat <- paste(shape_str, '\n', weight_str, sep = '')
            write(weight_file_dat, file = "weights_file.dat")

            f_max_file <- paste(lapply(f_max, function(fm) paste(fm, collapse = ' ')), collapse = ';')
            write(f_max_file, file = 'f_max_file.dat')

            Fin_file <- paste(lapply(Fin, function(i) paste(i, collapse = ' ')), collapse = '\n')
            write(Fin_file, file = 'input_ap.dat')

            Sys.sleep(0.1)
            prof_results <- system2("nvprof", args =  "./groovy_program", stdout = TRUE, stderr = TRUE)
            percent_kernel <- as.numeric(strsplit(strsplit(prof_results[grep('^ GPU', prof_results)], '\\s+')[[1]][4], '%')[[1]][1])
            print(percent_kernel)
            mean_percents <- mean_percents + percent_kernel / iters
        }
        params <- rbind(params, c(NNN, TTT))
        response <- c(response, mean_percents)
    }
}
colnames(params) <- c("Neurons", "Time")

save(params, response, file = "data/profiling_data.RData")
