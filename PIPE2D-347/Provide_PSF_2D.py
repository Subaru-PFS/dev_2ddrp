
import numpy as np

def provide_PSF_2D(x=None,y=None,PSF_version=None):
    """ Provides 2D PSF at any position in the detector plane
        This a version which takes a finite nubmer of pregenerated PSF and \
        creates the interpolated version at required position
        (Future: version which takes interpolated values for Zernike \
        coefficients and generates image on the fly?)
        
        To be used with the focused data taken on July 25 and 26
        (e.g., 21400 for HgAr, 21604 for Ne, 21808 for Kr)
        
        Example usage: ``provide_PSF_2D(10,2010)'' 10 is x-coordinate,
        and 2010 is y-coordinate        

    @param[in] x            x-coordinate
    @param[in] y            y-coordinate
    @param[in] PSF_version  version of the PSF input files
    @returns                numpy array, 100x100, oversampled 5 times, 
                            corresponding to 20x20 physical pixels
                            (300x300 microns)
    """    
    
    # on tiger the directory contaning array of PSFs is at:
    DATA_DIRECTORY='/tigress/ncaplar/PIPE2D-450/'
    
    if PSF_version is None:
        PSF_version='Sep12_v1'

    positions_of_simulation=np.load(DATA_DIRECTORY+\
                        'positions_of_simulation_00_from_'+PSF_version+'.npy')
    array_of_simulation=np.load(DATA_DIRECTORY+\
                        'array_of_simulation_00_from_'+PSF_version+'.npy')
    
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
        return array_of_simulation[index_of_closest_distance]
    
    # y-distance from the requested positions for all of the suitable simulated PSFs
    distances_of_y_requested_position_from_avaliable=\
    y-positions_of_simulation_in_acceptable_x_range[:,2]
    # out of the suitable PSFs which 2 are the closest
    index_of_1st_closest_simulated_psf=\
    np.where(np.abs(distances_of_y_requested_position_from_avaliable)==\
             np.sort(np.abs(distances_of_y_requested_position_from_avaliable))[0])[0][0]
    index_of_2nd_closest_simulated_psf=\
    np.where(np.abs(distances_of_y_requested_position_from_avaliable)==\
             np.sort(np.abs(distances_of_y_requested_position_from_avaliable))[1])[0][0]
    # where are these 2 closest PSF in the initial table
    index_of_1st_closest_simulated_psf_in_positions_of_simulation=\
    np.where(np.sum(positions_of_simulation,axis=1)==\
             np.sum(positions_of_simulation_in_acceptable_x_range[index_of_1st_closest_simulated_psf]))[0][0]
    index_of_2nd_closest_simulated_psf_in_positions_of_simulation=\
    np.where(np.sum(positions_of_simulation,axis=1)==\
             np.sum(positions_of_simulation_in_acceptable_x_range[index_of_2nd_closest_simulated_psf]))[0][0]
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
    
    # if you requested psf at the exact position of existing PSF use that one
    if y1_distance==0:
        return first_array_simulation
    else:    
        # create the predicted PSF as a linear interpolation of these two PSFs
        predicted_psf=(second_array_simulation-first_array_simulation*(y2_distance/y1_distance))/(1-y2_distance/y1_distance)
        return predicted_psf
    