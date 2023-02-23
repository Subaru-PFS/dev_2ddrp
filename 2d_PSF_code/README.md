# 2d_PSF_Code

Collection of files and documents related to 2D modeling of the point spread function in the Prime Focus Spectrograph pipelines.

- Tutorials: collection of code. It is made out of:
	- Cutting: examples showing how to cut the poststamp images from the full images
	- Radial_profile_estimation: how to estimate radial profile of an image
	- Zernike_multi_image_creation: create scripts to execute fitting routine 
	- Zernike_multi_image: analyze the results from the fitting routine (which fits many images at once)
	- Zernike_code: tutorial code which is able to generate single images with comments and examples

![Overview of the notebook](https://www.dropbox.com/s/53yqdv41yoomi2i/Screen%20Shot%202022-06-01%20at%207.38.34%20PM.png?raw=1)

- Zernike_Analysis_Module: for analysis of results from fitting routines
- Zernike_Cutting_Module: for cutting the post-stamp images of individual spots from full calexp data
- Zernike_Module: main python module for creating model point-spread functions and donuts
- Zernike_parameter_estimation: fitting routines to determine parameters describing individual spots
- Zernike_Residual_1D_Module: for plotting preliminary results in 1D