# Kernel K Means Clustering

## Overview
This project implements kernel K Means clustering on two dimensional data using an RBF kernel and visualizes the final clusters and centroids.

## Requirements
Install NumPy and Matplotlib.


## Data Format
Place the files `test_verileri_1.txt` and `test_verileri_2.txt` in the same directory as the script.  
Each file must contain two numeric columns separated by spaces, one sample per line.

## Configuration
You can change the following variables at the top of the script:

- `k` sets the number of clusters.  
- `var` sets the RBF kernel sigma value.  
- `input` selects which dataset is clustered.  
- `initMethod` can be `random`, `byCenterDistance`, or `byOriginDistance`.

## Main Functions
- `baslat` initializes cluster membership.  
- `rbfKernel` computes RBF kernel values.  
- `thirdTerm` computes internal cluster kernel averages.  
- `secondTerm` computes kernel distance between a point and a cluster.  
- `kernelKMeans` performs iterative kernel K Means until convergence.  
- `plot` visualizes clusters and centroids.

## Running
Save the script and data files in one folder, then run:

The algorithm runs and shows a plot of clusters and centroids.

## Output
- A Matplotlib figure showing final clusters.  
- `sonuc` contains clustered samples.  
- `centroid` contains centroid coordinates.  
- `iteration_counter` stores the number of iterations.

