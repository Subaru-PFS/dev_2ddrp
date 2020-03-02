"""
Created on Tue Jul 30 14:50:17 2019

@author: Neven Caplar
ncaplar@princeton.edu
www.ncaplar.com
 
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def find_centroid_of_flux(image):
    """
    function giving the position of weighted average of the flux in a square image
    
    @array[in] sci_image               numpy array with the values for the cutout of the science image (20x20 cutout)
    """
    
    
    x_center=[]
    y_center=[]

    I_x=[]
    for i in range(len(image)):
        I_x.append([i,np.sum(image[:,i])])

    I_x=np.array(I_x)

    I_y=[]
    for i in range(len(image)):
        I_y.append([i,np.sum(image[i])])

    I_y=np.array(I_y)


    x_center=(np.sum(I_x[:,0]*I_x[:,1])/np.sum(I_x[:,1]))
    y_center=(np.sum(I_y[:,0]*I_y[:,1])/np.sum(I_y[:,1]))

    return(x_center,y_center)



def residual_1D(sci_image,var_image,model_image):
    
    """
    
    @array[in] sci_image               numpy array with the values for the cutout of the science image (20x20 cutout)
    @array[in] var_image               numpy array with the cutout for the cutout of the variance image (20x20 cutout)
    @array[in] model_image             model (20x20 image)
     
    @array[out] init_lamda             1D extraction from the science image
    @array[out] std_init_lamda         error on the 1D extraction from the science image
    @array[out] init_removal_lamda     1D extraction from the residual image
    @array[out] std_init_removal_lamda error on 1D extraction from the residual image
    
    """
    assert sci_image.shape==(20,20)
    assert var_image.shape==(20,20)
    assert sci_image.shape==(20,20)  
    
    
    cental_pixel_for_x_value_int=int(round(find_centroid_of_flux(sci_image)[0]))
    
    multiplicative_factor_to_renormalize_to_40000=np.max(sci_image)/40000
    sci_image_smaller=sci_image[:,cental_pixel_for_x_value_int-3:cental_pixel_for_x_value_int+3]/multiplicative_factor_to_renormalize_to_40000
    var_image_smaller=var_image[:,cental_pixel_for_x_value_int-3:cental_pixel_for_x_value_int+3]/multiplicative_factor_to_renormalize_to_40000

    residual_initial_smaller=sci_image_smaller-model_image[:,cental_pixel_for_x_value_int-3:cental_pixel_for_x_value_int+3]/multiplicative_factor_to_renormalize_to_40000


    #################################
    # step 5 from Horne (http://adsabs.harvard.edu/abs/1986PASP...98..609H), very simplified
    inputimage_smaller=sci_image_smaller
    Px=np.sum(inputimage_smaller,axis=0)/np.sum(inputimage_smaller)
    var_inputimage_smaller=var_image_smaller
    #################################
    # Equation 8 from Horne with modification from Robert abut variance for extraction of signal
    # note that this uses profile from full thing, and not "residual profile"

    # nominator
    weighted_inputimage_smaller=inputimage_smaller*Px/(1)
    # denominator
    weights_array=np.ones((inputimage_smaller.shape[0],inputimage_smaller.shape[1]))*Px**2

    init_lamda=np.array(list(map(np.sum, weighted_inputimage_smaller)))/(np.array(list(map(np.sum,weights_array))))
    init_lamda_boxcar=np.array(list(map(np.sum, inputimage_smaller)))
    # Equation 8.5 from Horne
    var_f_std_lamda=1/np.sum(np.array(Px**2/(var_inputimage_smaller)),axis=1)
    std_init_lamda=np.sqrt(var_f_std_lamda)
    std_init_lamda_boxcar=np.sqrt(np.array(list(map(np.sum, var_inputimage_smaller))))


    #################################
    # Equation 8 from Horne with modification from Robert abut variance for initial removal
    # note that this uses profile from full thing, and not "residual profile"

    # nominator
    weighted_inputimage_smaller=residual_initial_smaller*Px/(1)
    # denominator
    weights_array=np.ones((residual_initial_smaller.shape[0],residual_initial_smaller.shape[1]))*Px**2

    init_removal_lamda=np.array(list(map(np.sum, weighted_inputimage_smaller)))/(np.array(list(map(np.sum,weights_array))))
    init_removal_lamda_boxcar=np.array(list(map(np.sum, residual_initial_smaller)))
    # Equation 8.5 from Horne
    var_init_removal_lamda=1/np.sum(np.array(Px**2/(var_inputimage_smaller)),axis=1)
    std_init_removal_lamda=np.sqrt(var_init_removal_lamda)
    return init_lamda,std_init_lamda,init_removal_lamda,std_init_removal_lamda

def chi_40000(sci_image,var_image,model_image):
    
    """
    crude algorithm to modify chi**2 which one would expect if the max flux of the science image was at 40000 counts
       
    
    @array[in] sci_image               numpy array with the values for the cutout of the science image (20x20 cutout)
    @array[in] var_image               numpy array with the cutout for the cutout of the variance image (20x20 cutout)
    @array[in] model_image             model (20x20 image)
    
    @float[out]                        chi**2
    
    """
    
    sci_image_renormalized,var_image_renormalized,model_image_renormalized=add_artificial_noise(sci_image,var_image,model_image)
    #multiplicative_factor_to_renormalize_to_40000=np.max(sci_image)/40000
    #sci_image_renormalized=sci_image/multiplicative_factor_to_renormalize_to_40000
    #var_image_renormalized=var_image/multiplicative_factor_to_renormalize_to_40000
    #model_image_renormalized=model_image/multiplicative_factor_to_renormalize_to_40000
    

    return np.mean((sci_image_renormalized-model_image_renormalized)**2/var_image_renormalized)


def add_artificial_noise(sci_image,var_image,model_image):
    
    """
    add extra noise so that it has comparable noise as if the max flux in the image (in the single pixel) is 40000
    
    @array[in] sci_image               numpy array with the values for the cutout of the science image (20x20 cutout)
    @array[in] var_image               numpy array with the cutout for the cutout of the variance image (20x20 cutout)
    @array[in] model_image             model (20x20 image)
    
    @array[out] sci_image              if max flux smaller than 40000, unchanged science image (20x20 cutout)
                                       if max flux larger than 40000, degraded science image
    @array[out]                        modified variance image
    @array[out] model_image            unchagned model (20x20 image)
    
    """

    # what is the ratio between the current science image and 40000 value 
    #multi_factor=np.max(sci_image)/40000
    
    # signal to noise ratio in the brightess pixel
    Max_SN_now=np.max(sci_image)/np.max(np.sqrt(var_image))
    
    # what is the ratio between the SN ratio in the brightest pixel to what I expect (which is roughly np.sqrt(40000/1.2)=220)
    # factor 1.2 in the previous line comes because variance is empirically a bit smaller than the signal
    dif_in_SN=Max_SN_now/220
    
    # prepare array which will contain artifically created noise
    artifical_noise=np.zeros_like(model_image)
    artifical_noise=np.array(artifical_noise)
    
    # minimal value in the variance image
    min_var_value=np.min(var_image)
    
    # for each pixel create additional artifical random noise, drawing from the normal distribution
    for i in range(len(artifical_noise)):
        for j in range(len(artifical_noise)):
            artifical_noise[i,j]=np.random.randn()*np.sqrt((dif_in_SN**2-1)*(var_image[i,j]-min_var_value))   
            
    # if you need to degrade image
    # return science image with additional noise
    # return variance image with stronger variance
    if dif_in_SN>1:        
        return (sci_image+artifical_noise),((dif_in_SN**2)*(var_image-min_var_value)+min_var_value),model_image
    else:
        # if you need to ``improve image''
        # return decreased variance image
        return (sci_image),((dif_in_SN**2)*(var_image-min_var_value)+min_var_value),model_image 


def plot_1D_residual(sci_image,var_image,model_image,title=None):
        
    """

    @array[in] sci_image               numpy array with the values for the cutout of the science image (20x20 cutout)
    @array[in] var_image               numpy array with the cutout for the cutout of the variance image (20x20 cutout)
    @array[in] model_image             model (20x20 image)
    @string[in] title                  custom title to appear above the plot


    @plot[out]                         diagnostic plot

    """
    
    
    init_lamda,std_init_lamda,init_removal_lamda,std_init_removal_lamda=residual_1D(sci_image,var_image,model_image)
    
    
    position_of_max_flux=np.where(init_lamda==np.max(init_lamda))[0][0]
    difference_from_max=range(20)-position_of_max_flux
    pixels_to_test=np.array(range(20))[(np.abs(difference_from_max)>2)&(np.abs(difference_from_max)<=6)]
    Q=np.mean(np.abs(init_removal_lamda[pixels_to_test]/std_init_removal_lamda[pixels_to_test]))
 
    plt.figure(figsize=(20,10))
    plt.errorbar(np.array(range(len(init_lamda))),init_lamda,yerr=std_init_lamda,fmt='o',elinewidth=2,capsize=12,markeredgewidth=2,label='data',color='orange')
    plt.errorbar(np.array(range(len(init_removal_lamda))),init_removal_lamda,yerr=std_init_removal_lamda,color='red',fmt='o',elinewidth=2,capsize=10,markeredgewidth=2,label='residual')

    for i in range(20):
        plt.text(-0.5+i, -1250, str("{:1.0f}".format(init_lamda[i])), fontsize=20,rotation=70.,color='orange')

    for i in range(20):
        plt.text(-0.5+i, -2050, str("{:1.1f}".format(init_removal_lamda[i]/std_init_removal_lamda[i])), fontsize=20,rotation=70.,color='red')
    
    if title is None:
        pass
    else:
        plt.title(str(title))
        
    plt.legend(loc=2, fontsize=22)
    plt.plot(np.zeros(20),'--',color='black')
    plt.ylim(-2500,2500)
    plt.ylabel('flux',size=25)
    plt.xlabel('pixel',size=25)
    plt.xticks(range(20))

    sci_image_40000,var_image_40000,model_image_40000=add_artificial_noise(sci_image,var_image,model_image)
    init_lamda,std_init_lamda,init_removal_lamda,std_init_removal_lamda=residual_1D(sci_image_40000,var_image_40000,model_image_40000)

    position_of_max_flux=np.where(init_lamda==np.max(init_lamda))[0][0]
    difference_from_max=range(20)-position_of_max_flux
    pixels_to_test=np.array(range(20))[(np.abs(difference_from_max)>2)&(np.abs(difference_from_max)<=6)]
    Q_40000=np.mean(np.abs(init_removal_lamda[pixels_to_test]/std_init_removal_lamda[pixels_to_test]))
    

    
    plt.text(19.5,2300, '$Q_{'+str(np.int(np.round(np.max(sci_image))))+'}$='+str("{:1.2f}".format(Q)),
            horizontalalignment='right',
            verticalalignment='top',fontsize=26)

    chi2=np.mean((model_image-sci_image)**2/var_image)

    plt.text(19.5,2000, '$\chi^{2}_{'+str(np.int(np.round(np.max(sci_image))))+'}$='+str("{:1.2f}".format(chi2)),
            horizontalalignment='right',
            verticalalignment='top',fontsize=26)

    chi2_40000=np.mean((model_image_40000-sci_image_40000)**2/var_image_40000)

    plt.text(19.5,1650, '$Q_{40000}$='+str("{:1.2f}".format(Q_40000)),
            horizontalalignment='right',
            verticalalignment='top',fontsize=26)
    plt.text(19.5,1300, '$\chi^{2}_{40000}$='+str("{:1.2f}".format(chi2_40000)),
            horizontalalignment='right',
            verticalalignment='top',fontsize=26)

    plt.axvspan(pixels_to_test[0]-0.5, pixels_to_test[3]+0.5, alpha=0.3, color='grey')
    plt.axvspan(pixels_to_test[4]-0.5, pixels_to_test[7]+0.5, alpha=0.3, color='grey')
    
def plot_1D_residual_custom(sci_image,var_image,model_image,title=None):
        
    """

    @array[in] sci_image               numpy array with the values for the cutout of the science image (20x20 cutout)
    @array[in] var_image               numpy array with the cutout for the cutout of the variance image (20x20 cutout)
    @array[in] model_image             model (20x20 image)
    @string[in] title                  custom title to appear above the plot


    @plot[out]                         diagnostic plot

    """
    
    
    init_lamda,std_init_lamda,init_removal_lamda,std_init_removal_lamda=residual_1D(sci_image,var_image,model_image)
    
    
    position_of_max_flux=np.where(init_lamda==np.max(init_lamda))[0][0]
    difference_from_max=range(20)-position_of_max_flux
    pixels_to_test=np.array(range(20))[(np.abs(difference_from_max)>2)&(np.abs(difference_from_max)<=6)]
    Q=np.mean(np.abs(init_removal_lamda[pixels_to_test]/std_init_removal_lamda[pixels_to_test]))
 
    plt.figure(figsize=(20,10))
    plt.errorbar(np.array(range(len(init_lamda)))[2:19],init_lamda[2:19],yerr=std_init_lamda[2:19],fmt='o',elinewidth=2,capsize=12,markeredgewidth=2,label='data',color='black')
    plt.errorbar(np.array(range(len(init_removal_lamda)))[2:19],init_removal_lamda[2:19],yerr=std_init_removal_lamda[2:19],color='red',fmt='o',elinewidth=2,capsize=10,markeredgewidth=2,label='residual')
    """
    for i in range(2,18):
        plt.text(-0.5+i, -1250, str("{:1.0f}".format(init_lamda[i])), fontsize=20,rotation=70.,color='orange')

    for i in range(2,18):
        plt.text(-0.5+i, -2050, str("{:1.1f}".format(init_removal_lamda[i]/std_init_removal_lamda[i])), fontsize=20,rotation=70.,color='red')
    
    if title is None:
        pass
    else:
        plt.title(str(title))
    """    
    plt.legend(loc=2, fontsize=30)
    plt.plot(np.zeros(20),'--',color='black')
    plt.ylim(-700,1500)
    plt.xlim(1.5,18.5)
    plt.ylabel('flux',size=35)
    plt.xlabel('pixel',size=35)
    plt.xticks(range(20)[2:19])

    sci_image_40000,var_image_40000,model_image_40000=add_artificial_noise(sci_image,var_image,model_image)
    init_lamda,std_init_lamda,init_removal_lamda,std_init_removal_lamda=residual_1D(sci_image_40000,var_image_40000,model_image_40000)

    position_of_max_flux=np.where(init_lamda==np.max(init_lamda))[0][0]
    difference_from_max=range(20)-position_of_max_flux
    pixels_to_test=np.array(range(20))[(np.abs(difference_from_max)>2)&(np.abs(difference_from_max)<=6)]
    Q_40000=np.mean(np.abs(init_removal_lamda[pixels_to_test]/std_init_removal_lamda[pixels_to_test]))
    

    """
    plt.text(19.5,2300, '$Q_{'+str(np.int(np.round(np.max(sci_image))))+'}$='+str("{:1.2f}".format(Q)),
            horizontalalignment='right',
            verticalalignment='top',fontsize=26)

    chi2=np.mean((model_image-sci_image)**2/var_image)

    plt.text(19.5,2000, '$\chi^{2}_{'+str(np.int(np.round(np.max(sci_image))))+'}$='+str("{:1.2f}".format(chi2)),
            horizontalalignment='right',
            verticalalignment='top',fontsize=26)

    chi2_40000=np.mean((model_image_40000-sci_image_40000)**2/var_image_40000)

    plt.text(19.5,1650, '$Q_{40000}$='+str("{:1.2f}".format(Q_40000)),
            horizontalalignment='right',
            verticalalignment='top',fontsize=26)
    plt.text(19.5,1300, '$\chi^{2}_{40000}$='+str("{:1.2f}".format(chi2_40000)),
            horizontalalignment='right',
            verticalalignment='top',fontsize=26)
    """
    plt.axvspan(pixels_to_test[0]-0.5, pixels_to_test[3]+0.5, alpha=0.3, color='grey')
    plt.axvspan(pixels_to_test[4]-0.5, pixels_to_test[7]+0.5, alpha=0.3, color='grey')
    
def plot_1D_residual_custom_large(sci_image,var_image,model_image,title=None):
        
    """

    @array[in] sci_image               numpy array with the values for the cutout of the science image (20x20 cutout)
    @array[in] var_image               numpy array with the cutout for the cutout of the variance image (20x20 cutout)
    @array[in] model_image             model (20x20 image)
    @string[in] title                  custom title to appear above the plot


    @plot[out]                         diagnostic plot

    """
    
    
    init_lamda,std_init_lamda,init_removal_lamda,std_init_removal_lamda=residual_1D(sci_image,var_image,model_image)
    
    
    position_of_max_flux=np.where(init_lamda==np.max(init_lamda))[0][0]
    difference_from_max=range(20)-position_of_max_flux
    pixels_to_test=np.array(range(20))[(np.abs(difference_from_max)>2)&(np.abs(difference_from_max)<=6)]
    Q=np.mean(np.abs(init_removal_lamda[pixels_to_test]/std_init_removal_lamda[pixels_to_test]))
    
    fig, ax = plt.subplots(figsize=[20, 10])

    ax.errorbar(np.array(range(len(init_lamda)))[2:19],init_lamda[2:19],yerr=std_init_lamda[2:19],fmt='o',elinewidth=2,capsize=12,markeredgewidth=2,label='data',color='black',ls='--')
    ax.errorbar(np.array(range(len(init_removal_lamda)))[2:19],init_removal_lamda[2:19],yerr=std_init_removal_lamda[2:19],color='red',fmt='o',elinewidth=2,capsize=10,markeredgewidth=2,label='residual')
    """
    for i in range(2,18):
        plt.text(-0.5+i, -1250, str("{:1.0f}".format(init_lamda[i])), fontsize=20,rotation=70.,color='orange')

    for i in range(2,18):
        plt.text(-0.5+i, -2050, str("{:1.1f}".format(init_removal_lamda[i]/std_init_removal_lamda[i])), fontsize=20,rotation=70.,color='red')
    
    if title is None:
        pass
    else:
        plt.title(str(title))
    """    
    ax.legend(loc=2, fontsize=35)
    ax.plot(np.zeros(20),'--',color='grey')
    ax.set_ylim(-10000,135000)
    ax.set_xticks(range(18))
    ax.set_xlim(1.5,18.5)
    ax.set_ylabel('flux',size=45)
    ax.set_xlabel('pixel',size=45)


    sci_image_40000,var_image_40000,model_image_40000=add_artificial_noise(sci_image,var_image,model_image)
    init_lamda,std_init_lamda,init_removal_lamda,std_init_removal_lamda=residual_1D(sci_image_40000,var_image_40000,model_image_40000)

    position_of_max_flux=np.where(init_lamda==np.max(init_lamda))[0][0]
    difference_from_max=range(20)-position_of_max_flux
    pixels_to_test=np.array(range(20))[(np.abs(difference_from_max)>2)&(np.abs(difference_from_max)<=6)]
    Q_40000=np.mean(np.abs(init_removal_lamda[pixels_to_test]/std_init_removal_lamda[pixels_to_test]))
    

    """
    plt.text(19.5,2300, '$Q_{'+str(np.int(np.round(np.max(sci_image))))+'}$='+str("{:1.2f}".format(Q)),
            horizontalalignment='right',
            verticalalignment='top',fontsize=26)

    chi2=np.mean((model_image-sci_image)**2/var_image)

    plt.text(19.5,2000, '$\chi^{2}_{'+str(np.int(np.round(np.max(sci_image))))+'}$='+str("{:1.2f}".format(chi2)),
            horizontalalignment='right',
            verticalalignment='top',fontsize=26)

    chi2_40000=np.mean((model_image_40000-sci_image_40000)**2/var_image_40000)

    plt.text(19.5,1650, '$Q_{40000}$='+str("{:1.2f}".format(Q_40000)),
            horizontalalignment='right',
            verticalalignment='top',fontsize=26)
    plt.text(19.5,1300, '$\chi^{2}_{40000}$='+str("{:1.2f}".format(chi2_40000)),
            horizontalalignment='right',
            verticalalignment='top',fontsize=26)
    """
    #ax.axvspan(pixels_to_test[0]-0.5, pixels_to_test[3]+0.5, alpha=0.3, color='grey')
    #ax.axvspan(pixels_to_test[4]-0.5, pixels_to_test[7]+0.5, alpha=0.3, color='grey')
    
    axins = inset_axes(ax, width="100%", height="100%", loc=1,bbox_to_anchor=(0.63,0.55,0.35,0.38), bbox_transform=ax.transAxes)
    axins.errorbar(np.array(range(len(init_lamda)))[2:19],init_lamda[2:19],yerr=std_init_lamda[2:19],fmt='o',elinewidth=2,capsize=12,markeredgewidth=2,label='data',color='black',ls='--')
    axins.errorbar(np.array(range(len(init_removal_lamda)))[2:19],init_removal_lamda[2:19],yerr=std_init_removal_lamda[2:19],color='red',fmt='o',elinewidth=2,capsize=10,markeredgewidth=2,label='residual')
    axins.set_ylim(-400,1400)
    axins.set_xlim(1.5,18.5)
    axins.plot(np.zeros(20),'--',color='grey')
    axins.axvspan(pixels_to_test[0]-0.5, pixels_to_test[3]+0.5, alpha=0.3, color='grey')
    axins.axvspan(pixels_to_test[4]-0.5, pixels_to_test[7]+0.5, alpha=0.3, color='grey')
    
    fig.savefig('/Users/nevencaplar/Documents/PFS/Poster/Poster2019/' + '1d.png', bbox_inches='tight')