import numpy as np

def provide_PSF_2D(x=None,y=None,PSF_version=None):
    """ Provide 2D PSF at any position in the detector plane
        This a version which takes a finite nubmer of pregenerated PSF and \
        creates the interpolated version at required position
        (Future: version which takes interpolated values for Zernike \
        coefficients and generates image on the fly?)
        
    This version with      
        

    @param[in] x            x-coordinate
    @param[in] y            y-coordinate
    @param[in] PSF_version  version of the PSF input files
    
    @returns                numpy array, (if PSF input file is  Apr15_v2 is 189x189, oversampled 9 times, \
                            corresponding to 21x21 physical pixels (315x315 microns))
    """    
    
    # on tiger this directory is at:
    #DATA_DIRECTORY='/tigress/ncaplar/PIPE2D-521/'
    DATA_DIRECTORY='/Users/nevencaplar/Documents/PFS/Tickets/PIPE2D-521/'
    
    if PSF_version is None:
        PSF_version='Apr15_v2'

    positions_of_simulation=np.load(DATA_DIRECTORY+'positions_of_simulation_00_from_'+PSF_version+'.npy',allow_pickle=True)
    array_of_simulation=np.load(DATA_DIRECTORY+'array_of_simulation_00_from_'+PSF_version+'.npy',allow_pickle=True)
    
    # x and y position with simulated PSFs
    x_positions_of_simulation=positions_of_simulation[:,1]
    y_positions_of_simulation=positions_of_simulation[:,2]
    
    # This is a simple code that finds the closest avaliable PSFs, given the x and y position
    # This will have to be improved in order when we get to work with the full populated dectector plane
    
    
    # how far in x-dimension are you willing to search for suitable simulated PSFs
    x_search_distance=20
    # positions of all simulated PSFs in that range
    positions_of_simulation_in_acceptable_x_range=\
    positions_of_simulation[(x_positions_of_simulation<(x+x_search_distance))\
                            &(x_positions_of_simulation>(x-x_search_distance))]
    
    # if there are no simulated PSF avaliable in the specified x-range we are not able to provide the solution
    if len(positions_of_simulation_in_acceptable_x_range)<2:
        print('No simulated PSFs are avaliable in this x-area of the detector,')
        print('probably because this fiber has not been illuminated;')
        print('returning the closest avaliable PSFs, BUT that is probably not what you want')
        distances=np.sqrt(((x-x_positions_of_simulation)**2+\
                           (y-y_positions_of_simulation)**2).astype(float))
        index_of_closest_distance=np.where(distances[distances==\
                                                     np.min(distances)])[0][0]
        
        # ! change here on April 22, 2020!
        #       changed so the output is more similar (output as a list) to more general case below
        return [array_of_simulation[index_of_closest_distance]]
        # ! end of change here on April 22, 2020
    
    # y-distance from the requested positions for all of the suitable simulated PSFs
    distances_of_y_requested_position_from_avaliable=\
    y-positions_of_simulation_in_acceptable_x_range[:,2]
    
    # ! change here on April 22, 2020!
    #    separate the distances into distances which are for the spots above the requested position and below the requested position
    distances_of_y_requested_position_from_avaliable_which_are_above_the_spot=\
    distances_of_y_requested_position_from_avaliable[distances_of_y_requested_position_from_avaliable<0]
    distances_of_y_requested_position_from_avaliable_which_are_below_the_spot=\
    distances_of_y_requested_position_from_avaliable[distances_of_y_requested_position_from_avaliable>0]
    
    #    if there are no sources above, do not do interpolation and select the nearest source
    if len(distances_of_y_requested_position_from_avaliable_which_are_above_the_spot)==0:
        index_of_1st_closest_above_simulated_psf=-99 
    else:
        index_of_1st_closest_above_simulated_psf=\
        np.where(distances_of_y_requested_position_from_avaliable==\
                 np.max(distances_of_y_requested_position_from_avaliable_which_are_above_the_spot))[0][0]    

    #    if there are no sources below, do not do interpolation and select the nearest source
    if len(distances_of_y_requested_position_from_avaliable_which_are_below_the_spot)==0:
        index_of_1st_closest_below_simulated_psf=-99
    else:
        index_of_1st_closest_below_simulated_psf=\
        np.where(distances_of_y_requested_position_from_avaliable==\
                 np.min(distances_of_y_requested_position_from_avaliable_which_are_below_the_spot))[0][0]       

    if index_of_1st_closest_below_simulated_psf==-99:
        index_of_1st_closest_below_simulated_psf=index_of_1st_closest_above_simulated_psf
    if index_of_1st_closest_above_simulated_psf==-99:
        index_of_1st_closest_above_simulated_psf=index_of_1st_closest_below_simulated_psf
    # ! end of change here on April 22, 2020
    
    # where are these 2 closest PSF in the initial table
    index_of_1st_closest_simulated_psf_in_positions_of_simulation=\
    np.where(np.sum(positions_of_simulation,axis=1)==\
             np.sum(positions_of_simulation_in_acceptable_x_range[index_of_1st_closest_above_simulated_psf]))[0][0]
    index_of_2nd_closest_simulated_psf_in_positions_of_simulation=\
    np.where(np.sum(positions_of_simulation,axis=1)==\
             np.sum(positions_of_simulation_in_acceptable_x_range[index_of_1st_closest_below_simulated_psf]))[0][0]



    # extract the 2 simulated PSFs
    first_array_simulation=\
    array_of_simulation[index_of_1st_closest_simulated_psf_in_positions_of_simulation]
    second_array_simulation=\
    array_of_simulation[index_of_2nd_closest_simulated_psf_in_positions_of_simulation]
        
    # distance of each PSF from the proposed position

    y1_distance=\
    y-positions_of_simulation[index_of_1st_closest_simulated_psf_in_positions_of_simulation][2]
    y2_distance=\
    y-positions_of_simulation[index_of_2nd_closest_simulated_psf_in_positions_of_simulation][2]
    
    # ! change here on April 22, 2020!
    #       if you requested psf at the exact position of existing PSF use that one OR
    #       if you are outside of the range covered by spots, use the last avaliable image
    if y1_distance==0 or y1_distance==y2_distance:


        #    changed so the output is equivalent as in more general case
        return first_array_simulation,first_array_simulation,second_array_simulation,y1_distance,y2_distance
        # ! end of change here on April 22, 2020
    else:    
        # create the predicted PSF as a linear interpolation of these two PSFs
        predicted_psf=(second_array_simulation-first_array_simulation*(y2_distance/y1_distance))/(1-y2_distance/y1_distance)
        return predicted_psf,first_array_simulation,second_array_simulation,y1_distance,y2_distance
    
    