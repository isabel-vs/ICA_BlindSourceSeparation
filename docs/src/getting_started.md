# Getting started
This package provides multiple algorithms for Blind Source Separation. 
At this time the user can choose between JADE, Shibbs and Picard. 

## How to use the datatype SensorData
Key to using the package is the SensorData datatype. The user can either create a SensorData instance by loading it from disk using the read_dataset() function, or he can construct his own by provididing an array of timestamps of length N and a corresponding matrix of size NxM, where M is the amount of sensors in the dataset.
When the user has created a SensorData instance, he might plot it or perform the Blind Source separation using the perform_separation function.

## Code Example 
Load a dataset from disk

    x = read_dataset("data/foetal_ecg.dat")

Plot dataset

    plot_dataset(x)

Prepare JADE algorithm

    algo = Jade(2)

Prepare Shibbs algorithm

    algo = Shibbs(2, 1000)

Prepare Picard algorithm

    algo = Picard(3, 200, 1e-6, 1e-2, 10, true)

Run source separation. This yields the separated dataset as well as the unmixing matrix

    x, s = perform_separation(x, algo)

### Complete example:

This example plots the original whitened data, as well as the results of Jade, Shibbs and Picard algorithm.

    x = read_dataset("data/foetal_ecg.dat")
    plot_dataset(x)

    algo = Jade(2)
    a, _ = perform_separation(x, algo)
    plot_dataset(a)

    algo = Shibbs(2, 1000)
    b, _ = perform_separation(x, algo)
    plot_dataset(b)

    algo = Picard(3, 200, 1e-6, 1e-2, 10, true)
    c, _ = perform_separation(x, algo)
    plot_dataset(c)
