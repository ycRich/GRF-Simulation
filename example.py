from GRF import *
############# example ##############
# specify a uniform sdf function
# start, end - uniformly distributed within [start, end]
# range      - the range of the sdf function 
# num        - the number of points used to describe the sdf, or resolution
# the GRF constructed will have resolution 2*(num-1) by 2*(num-1)

sdf = sdf_uniform(start = 10, end = 20, range = [0,100], num = 101)

# sdf is the spectral density function of the GRF
# number_of_points specifies the number of random variables generated during the process
# large number_of_points result in more accurate construction but increases computational costs
image = gaussianRandomField(sdf, number_of_points = 500)
plt.figure('GRF')
plt.imshow(image, cmap = 'hot')
plt.show()