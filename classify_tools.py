import numpy as np 
import numpy.ma as ma
from scipy import signal 
import os
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.colors import ListedColormap
from osgeo import gdal
import whitebox


def read_geotiff(filename):
    """
    Read in a geotiff with gdal and return numpy array and gdal dataset object 

    Parameters
    ---------
    filename: full filename (.tif) (string)

    Returns
    ---------
    arr: numpy array of 1st band of raster (numpy array)
    ds: GDAL Dataset object

    """
    ds = gdal.Open(filename)
    arr = ds.GetRasterBand(1).ReadAsArray()
    return arr, ds


def write_geotiff(filename, arr, in_ds, nullval):
    """
    Write numpy array to geotiff with GDAL

    Parameters
    -------
    filename: output filename (string)
    arr: numpy array to be exported to a geotiff file (numpy array)
    in_ds: GDAL Dataset object which includes all metadata (GDAL dataset object)

    """
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(nullval)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)



def get_raster_extent(src):
    """
    This function gets the raster extent and outputs both the bounds and
    the coordinates of the four corners

    Parameters
    ----------
    src : the GDAL dataset object for the desired raster

    Returns
    -------
    geometry : the coordinates of the four corners of the raster extent (list)
    bounds : the bounds of the raster [minx, miny, maxx, maxy] (list)

    """
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    geometry = [[ulx,lry], [ulx,uly], [lrx,uly], [lrx,lry]]
    bounds = [ulx, lry, lrx, uly]
    
    return geometry, bounds

    

def generate_terrainparams(file, coarsen=None, plot_terrain=True):
    """
    This function uses the Whitebox toolkit to generate the terrain parameters
    needed for classifying landforms from the DEM. These include slope,
    plan curvature, minimum curvature, and maximum curvature.

    Parameters
    ----------
    file : full file path (string)
    coarsen : option to coarsen the DEM before calculating terrain parameters,
            this is used for the mean aggregation scaling option
    plot : option to plot the results (default is True)


    Returns
    -------
    See WhiteboxTools user manual for more info (https://www.whiteboxgeo.com/manual/wbt_book/)
    slope : the slope gradient (degrees)(numpy array)
    plancurv : curvature of a contour line at a given point on the
                topographic surface, positive indicates flow divergence and
                negative indicates flow convergence (numpy array)
    mincurv : curvature of a principal section with the lowest value of
                curvature at a given point of the topographic surface (numpy array)
    maxcurv : curvature of a principal section with the highest value of
                curvature at a given point of the topographic surface (numpy array)

    """
    
    name= os.path.splitext(os.path.basename(file))[0]
    direc = os.path.dirname(file)
    
    # Load whitebox tools and set working directory
    wbt = whitebox.WhiteboxTools()
    wbt.set_working_dir(direc)
    wbt.set_verbose_mode(False)  # if you want to print everything out, switch to True
    
    if coarsen is not None:
        name = name+'_'+str(coarsen)+'agg'
        wbt.aggregate_raster(file,(name+'.tif'),agg_factor=coarsen,type='mean')
        file = (name+'.tif')
    
    
    # Using whitebox, calculate slope, planform curvature, minimal curvature, and maximum curvature
    wbt.slope(file,(name +'_slope.tif'))
    wbt.plan_curvature(file,(name + '_plancurvature.tif'), log=False, zfactor=None)
    wbt.minimal_curvature(file,(name + '_mincurvature.tif'), log=False, zfactor=None)
    wbt.maximal_curvature(file,(name + '_maxcurvature.tif'), log=False, zfactor=None)

    # Load these terrain parameter rasters in as arrays
    dem_coarse = iio.imread(os.path.join(direc,file))
    slope = iio.imread(os.path.join(direc,(name+'_slope.tif')))
    plancurv = iio.imread(os.path.join(direc,(name+'_plancurvature.tif')))
    mincurv = iio.imread(os.path.join(direc,(name+'_mincurvature.tif')))
    maxcurv = iio.imread(os.path.join(direc,(name+'_maxcurvature.tif')))
    
    if plot_terrain is True:
        # plot terrain parameters as a check
        fig, axs = plt.subplots(2,2)
        plt.tight_layout()
        axs[0,0].imshow(slope)
        axs[0,0].set_title('Slope')
        axs[0,1].imshow(plancurv,vmin=np.mean(plancurv)-np.std(plancurv)*2,vmax=np.mean(plancurv)+np.std(plancurv)*2)
        axs[0,1].set_title('Plan Curvature')
        axs[1,0].imshow(mincurv,vmin=np.mean(mincurv)-np.std(mincurv)*3,vmax=np.mean(mincurv)+np.std(mincurv)*3)
        axs[1,0].set_title('Minimal Curvature')
        axs[1,1].imshow(maxcurv,vmin=np.mean(maxcurv)-np.std(maxcurv)*3,vmax=np.mean(maxcurv)+np.std(maxcurv)*3)
        axs[1,1].set_title('Maximal Curvature')
        
    return slope, plancurv, mincurv, maxcurv, dem_coarse



def landform_class(slope,plancurv,mincurv,maxcurv,Tslope,Tcurve):
    """
    This function classifies each pixel into a landform type according to
    Table 5.3 in Wood (1996).
    
    Wood, J. (1996). The geomorphological characterisation of digital
    elevation models. University of Leicester (United Kingdom).

    Parameters
    ----------
    slope : the slope gradient (degrees)(numpy array)
    plancurv : curvature of a contour line at a given point on the
                topographic surface, positive indicates flow divergence and
                negative indicates flow convergence (numpy array)
    mincurv : curvature of a principal section with the lowest value of
                curvature at a given point of the topographic surface (numpy array)
    maxcurv : curvature of a principal section with the highest value of
                curvature at a given point of the topographic surface (numpy array)
    Tslope : slope tolerance, or the minimum slope that represents a true slope (float)
    Tcurve : curvature tolerance, or minimum plan curvature that represents
                a true cross-sectional convexity/concavity (float)

    Returns
    -------
    featrs : landform classification array with pixels classified as planar, pit,
                channel, pass, ridge, or peak (numpy array)

    """
    # 1)	Planar
    # 2)	Pit
    # 3)	Channel
    # 4)	Pass (saddle)
    # 5)	Ridge
    # 6)	Peak

    featrs = np.empty((slope.shape))*np.nan

    featrs[np.where((mincurv < -Tcurve) & (maxcurv < -Tcurve))]= 2
    featrs[np.where((mincurv < -Tcurve) & (maxcurv >= Tcurve))]= 3
    featrs[np.where((maxcurv > Tcurve) & (mincurv > Tcurve))]= 6
    featrs[np.where((maxcurv > Tcurve) & (mincurv < -Tcurve))]= 4      
    featrs[np.where((maxcurv > Tcurve) & (np.abs(mincurv) < Tcurve))]= 5 
    featrs[np.where((slope > Tslope) & (plancurv > Tcurve))]= 5
    featrs[np.where((slope > Tslope) & (plancurv < -Tcurve))]= 3
    featrs[np.where((slope > Tslope) & (np.abs(plancurv) < Tcurve))]= 1
    featrs[np.isnan(featrs)]=1
    
    featrs[np.isnan(slope)]=np.nan
    
    return featrs


def colorbar_index(ax, ncolors, cmap):
    """
    Creates a discrete colorbar with tick marks and labels vertically centered
    in patch.

    Parameters
    ----------
    ax : plot axes object
    ncolors : number of discrete colors needed for plotting
    cmap : colormap

    Returns
    -------
    cbar : colorbar object

    """
    # cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(0.5, ncolors+0.5)
    cbar = plt.colorbar(mappable,ax=ax)
    cbar.set_ticks(np.linspace(1, ncolors, ncolors))
    
    return cbar



def plot_landforms(featrs, nullval):
    """
    Mask NoData values and plot landform classification array for visualization

    Parameters
    ----------
    featrs : landform classification array (numpy array)
    nullval : NoData value
    
    """
    
    # colors = ['w','k','b','g','y','r']
    # cmap_terr = ListedColormap(colors)
    landclass = {nullval: 'NoData', 1:'Planar',2:'Pit',3:'Channel',4:'Pass',5:'Ridge',6:'Peak'}
    # color_map = {1:'w', 2:'k', 3:'b', 4:'g', 5:'y', 6:'r', nullval: 'w'}
    
    # define color map  for array
    color_map = {nullval: (211, 211, 211), # light gray
                  1: (255, 255, 255), # white
                  2: (0, 0, 0), # black
                  3: (0, 0, 255), # blue
                  4: (0, 127, 0), # green
                  5: (191, 191, 0), # yellow
                  6: (255, 0, 0), # red
                  } # white
    
    # define color map for colorbar
    cmap_terr = {nullval: colors.to_rgb('lightgray'), # light gray
                 1: colors.to_rgb('w'), # white
                 2: colors.to_rgb('k'), # black
                 3: colors.to_rgb('b'), # blue
                 4: colors.to_rgb('g'), # green
                 5: colors.to_rgb('y'), # yellow
                 6: colors.to_rgb('r')} # red
    
    cmap_terr = ListedColormap(list(cmap_terr.values()))
    
    # make a 3d numpy array that has a color channel dimension   
    data_3d = np.ndarray(shape=(featrs.shape[0], featrs.shape[1], 3), dtype=int)
    for i in range(0, featrs.shape[0]):
        for j in range(0, featrs.shape[1]):
            data_3d[i][j] = color_map[featrs[i][j]]
    
    fig, ax = plt.subplots(1)
    featrs = ma.masked_where(featrs==nullval,featrs)
    plt.imshow(data_3d)  # problem, if there is one class missing from this figure, the colorbar is wrong
    cbar = colorbar_index(ax,len(color_map),cmap_terr)
    cbar.set_ticklabels(landclass.values())
    plt.show()
    


def fit_local_quad(img, dx, dy, window):
    """
    Estimates local quadratic approximations of a DEM img.  Consider a WxW window of the image
    with the point (0,0) at center of the pixel in the middle of this window.  A quadratic 
    approximate of the DEM around this pixel can be constructed with the form   

    f(x) = ax^2 + by^2 + cxy + dx + ey + f

    Where a,b,c,d,e,f are coefficients.    This function computes these coefficients for all windows
    in the image. 

    Arguments
    ---------
    img: np.array
        An Ny by Nx matrix containing elevations.
    dx: float
        The constant pixel size in the x direction. Typically in units of meters, but it doesn't matter as 
        long as the units of dx and dy are consistent.
    dy: float 
        The constant pixel size in the y direction.  Typically in units of meters, but it doesn't matter as 
        long as the units of dx and dy are consistent.
    window: int
        The number of pixels that define the window.  This must be odd.

    Returns
    --------
    coeffs: list[np.array]
        A list of 2d arrays containing the coefficients [a,b,c,d,e,f].  Each array is the same shape as the 
        DEM image.

    """

    # Window size must be odd 
    if (window%2)!=1:
        raise ValueError('Window size must be odd.')
    
    local_xs = dx*(np.arange(window) - (window-1)/2)
    local_ys = dy*(np.arange(window) - (window-1)/2)

    local_X, local_Y = np.meshgrid(local_xs,local_ys)

    # Construct the vandermonde matrix
    Vand = np.hstack([ local_X.reshape(-1,1)*local_X.reshape(-1,1), 
                       local_Y.reshape(-1,1)*local_Y.reshape(-1,1),
                       local_X.reshape(-1,1)*local_Y.reshape(-1,1),
                       local_X.reshape(-1,1),
                       local_Y.reshape(-1,1),
                       np.ones((window*window,1))])
    
    # Compute the pseudo-inverse from the SVD
    U,S,V = np.linalg.svd(Vand, full_matrices=False)
    pinv = V.T@np.diag(1.0/S)@U.T
    
    # Use correlations of the pseudo inverse with the image to compute the coefficients
    coeffs = [signal.correlate(img, pinv[i,:].reshape(window,window), mode='same', method='fft') for i in range(6)]
    
    return coeffs
    
    
if __name__=='__main__':

    Nx = 128
    Ny = 256
    
    dx = 10.0
    dy = 10.0

    x = dx*np.arange(Nx)
    y = dy*np.arange(Ny)

    X, Y = np.meshgrid(x,y)

    test_img = X*X + Y*Y

    window = 9
    coeffs = fit_local_quad(test_img, dx, dy, window)

