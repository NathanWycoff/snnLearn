# snnLearn - Spiking Neural Network Simulation for Machine Learning

Professor Warburton,

This is what will someday be an R package. I will be writing the computational pieces in Cpp.

Instead of building my code manually, you will have R install them for you. For this to work, you will need to install R. 

# Installation

First, if you're on Mac, make sure you have xcode developer tools installed. Run 

```
bash > xcode-select --install
```

if you don't.

Next, install R (brew install r).

Then, install the devtools package:

```
bash > R
<<welcome message>>
R > install.packages('devtools')
```

then use that to install my package:

```
bash > git clone https://github.com/NathanWycoff/snnLearn.git
bash > cd snnLearn
bash > R
<<welcome message>>
R > devtools::install('.')
```

Then, you can load the function using 'require(snnLearn)'.

This package is harder to install than usual because it requires compilation and is not yet hosted on CRAN (which keeps binaries available for Windows, OSX, and Ubuntu).

Do '?goc' after leading snnLearn for details.

The "goc" function calls a Cpp "intermediary" function which then calls pure Cpp code. The code may be found in src/layer_lif.cpp. The Cpp code uses armadillo for matrix multiplication. The appropriate headers should be automatically installed during installation of my package.

# Example

Try running this code in an R terminal.

```
require(snnLearn)

## A LIF with layer structure
learn_rate <- 0.1

t_eps <- 0.01
t_end <- 10
ts <- seq(0, t_end, by = t_eps)
t_steps <- length(ts)

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

# Simulate the Network, print out firing times of each layer
l_h <- length(n_h)
goc(n_in, l_h, n_h, n_out, Ws, Fin, t_eps, t_steps) 
```
