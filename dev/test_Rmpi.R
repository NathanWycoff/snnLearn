#!/usr/bin/Rscript
#  dev/test_Rmpi.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.29.2018

require('Rmpi')

mpi.spawn.Rslaves(nslaves = 10)

n <- 10
mpi.remote.exec(n)

mpi.quit()
