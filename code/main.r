# Load necessary libraries
library(akima)
library(lattice)

# Load the data
data <- read.table("Group_1_Data.txt", header = FALSE, col.names = c("x", "y", "z"))

# Create an interpolation grid
x <- seq(min(data$x), max(data$x), length.out = 100)
y <- seq(min(data$y), max(data$y), length.out = 100)
grid <- expand.grid(x = x, y = y)

# Interpolate the z values
library(MBA)
z <- interp(data$x, data$y, data$z, grid$x, grid$y)

# Plot the original data points
plot(data$x, data$y, main = "Original Data Points")

# Plot the interpolated surface
persp(grid$x, grid$y, matrix(z, nrow = 100), main = "Interpolated Surface")