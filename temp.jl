using Plots
using Statistics
using LinearAlgebra
using ICA_BlindSourceSeparation

n_sources = 2
n_samples = 1000
# Create a time vector for our signals
time = range(0, stop=10, length=n_samples)

# Source 1: A sine wave
s1 = sin.(0.8 * 2 * π * time)' # Note the transpose ' to make it a 1x1000 row vector

# Source 2: A square wave
s2 = sign.(sin.(1.5 * 2 * π * time))' # Transpose to a 1x1000 row vector

S_true = vcat(s1, s2)

# mixing matrix
A = [0.8 0.5; 0.3 0.9] # Using a fixed matrix

# Mix the sources
X = (A * S_true)'

mixed = sensorData(time, X)

separated, V = perform_separation(mixed, Jade(2))

p1 = plot(time, S_true', title="Original Sources (S)", label=["Sine" "Square"], lw=2)
p2 = plot(time, X, title="Mixed Signals (X)", label=["Mixture 1" "Mixture 2"], lw=2)
p3 = plot(time, separated.data, title="Recovered Signals (Y)", label=["Estimate 1" "Estimate 2"], lw=2)

plot(p1, p2, p3, layout=(3, 1), size=(800, 600), legend=:outertopright)