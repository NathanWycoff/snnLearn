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

then use that to install my package and its dependencies:

```
bash > R
<<welcome message>>
R > devtools::install_github('NathanWycoff/snnLearn')
```

Then, you can load the function using 'require(snnLearn)'.

This package is harder to install than usual because it requires compilation and is not yet hosted on CRAN (which keeps binaries available for Windows, OSX, and Ubuntu).

Do '?goc' after leading snnLearn for details.

The "goc" function calls a Cpp "intermediary" function which then calls pure Cpp code. The code may be found in src/layer_lif.cpp. The Cpp code uses Armadillo for matrix multiplication. The appropriate headers should be automatically installed during installation of my package.

Don't hesitate to email me if something here breaks down (I've tested that these instructions work on a Mac).

# Example

Try running this code in an R terminal.

```
require(snnLearn)
example(goc)
````
