#!/usr/bin/Rscript
#  dev/analyze_profiling.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.04.2018

# Profiling data
load("data/profiling_data.RData")

png("images/perc_vs_etc.png", width = 1000)
par(mfrow=c(1,2))
plot(response ~ params[,1], main = "Percent in Kernel versus Neuron Count", xlab = "Neurons", ylab = "Percent of Time")
plot(response ~ params[,2], main = "Percent in Kernel versus Execution Time", xlab = "Execution Time", ylab = "Percent of Time")
dev.off()

# Timing data
load("data/large_timing.RData")

png("images/exec_time.png", width = 480)
plot(times ~ Ns[1:length(times)], main = "Execution Time versus Network Width", xlab = "Neurons", ylab = "Time")
dev.off()
