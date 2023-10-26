import numpy as np
import numpy.ma as ma
import os
import matplotlib.pyplot as plt
from osgeo import gdal, osr
from scipy import signal
from scipy import stats
from datetime import datetime
import whitebox
import yaml

import classify_tools as ct



def preprocess_dem(dem_file, mask_file, nullval, dst_epsg, dst_resoln, mask=True, plot=True):
    """
    This function reads in the raw dem and mask files and reprojects, resamples,
    and clips the rasters, as needed. Lastly, it masks the DEM, setting all
    pixels that are not land ice to the null (NoData) value provided.

    Parameters
    ----------
    dem_file : full file path to DEM geotiff file (string)
    mask_file : full file path to land ice mask geotiff file (string), where 1 is land ice and 0 is all else
    nullval : value to input for NoData pixels (float)
    dst_epgs : EPSG number for target coordinate reference system (int)
    dst_resoln: target resolution (int)

    Returns
    -------
    out_file : full file path to masked DEM geotiff file (string)

    """
    
    # dem directory
    direc = os.path.dirname(dem_file)
    
    # load dem as GDAL dataset object and array
    dem, ds = ct.read_geotiff(dem_file)
    
    # get target CRS info
    dst_crs = 'EPSG:' + str(dst_epsg)
    dstcrs = osr.SpatialReference()
    dstcrs.ImportFromEPSG(dst_epsg)
    
    if mask == True:
        # load mask as GDAL dataset object and array
        maskarr, dsm = ct.read_geotiff(mask_file)
        
        # check if mask and dem have same CRS
        crsdem = osr.SpatialReference()
        crsdem.ImportFromWkt(ds.GetProjection())
        crsmask = osr.SpatialReference()
        crsmask.ImportFromWkt(dsm.GetProjection())
        
        # reproject DEM or mask, if needed
        if (crsdem.IsSame(dstcrs) == 0):
            print('Reprojecting DEM to ' + dst_crs)
            reproj_ds = os.path.join(direc,'dem_reproject.tif')
            ds = gdal.Warp(reproj_ds, ds, dstSRS = dst_crs, xRes = dst_resoln, yRes = dst_resoln, dstNodata = nullval, resampleAlg = 'bilinear')
            dem = ds.GetRasterBand(1).ReadAsArray()
        if (crsmask.IsSame(dstcrs) == 0):
            print('Reprojecting mask to ' + dst_crs)
            reproj_dsm = os.path.join(direc,'mask_reproject.tif')
            dsm = gdal.Warp(reproj_dsm, dsm, dstSRS = dst_crs, xRes = dst_resoln, yRes = dst_resoln, dstNodata = nullval, resampleAlg = 'mode')
            maskarr = dsm.GetRasterBand(1).ReadAsArray()
        
        # check if mask and dem have same CRS
        crsdem = osr.SpatialReference()
        crsdem.ImportFromWkt(ds.GetProjection())
        crsmask = osr.SpatialReference()
        crsmask.ImportFromWkt(dsm.GetProjection())
        if (crsdem.IsSame(crsmask) == 0):
            print('Mask and DEM CRSs do not match')
            
            
        _, dem_resoln, _, _, _, _ = ds.GetGeoTransform()
        _, mask_resoln, _, _, _, _ = dsm.GetGeoTransform()
        
        # if mask is significantly larger than DEM, do initial rough clip to reduce resampling time and file size
        # the final clip will be done after resampling
        if (maskarr.shape[0] - dem.shape[0] > 50) or (maskarr.shape[1] - dem.shape[1] > 50):
            dem_extent, dem_bounds = ct.get_raster_extent(ds)
            print(dem_bounds)
            inc = [dem_resoln*-3, dem_resoln*-3, dem_resoln*3, dem_resoln*3]
            dem_bounds = [sum(i) for i in zip(dem_bounds,inc)]
            dsm = gdal.Warp(os.path.join(direc,'mask_reduced.tif'), dsm, outputBounds = tuple(dem_bounds))
            maskarr = dsm.GetRasterBand(1).ReadAsArray()
        
            
        ## check if mask and dem have the same resolution and resample if not
        if dem_resoln is not dst_resoln:
            print('Resampling DEM to ' + str(dst_resoln))
            resamp_ds = os.path.join(direc,'dem_resample.tif')
            ds = gdal.Warp(resamp_ds, ds, xRes=dst_resoln, yRes=dst_resoln, resampleAlg='bilinear')
            dem = ds.GetRasterBand(1).ReadAsArray()
        if mask_resoln is not dst_resoln:
            print('Resampling mask to ' + str(dst_resoln))
            resamp_dsm = os.path.join(direc,'mask_resample.tif')
            dsm = gdal.Warp(resamp_dsm, dsm, xRes=dst_resoln, yRes=dst_resoln, resampleAlg='mode')
            maskarr = dsm.GetRasterBand(1).ReadAsArray()
        
        # clip mask
        if (dem.shape[0] < maskarr.shape[0]) and (dem.shape[1] < maskarr.shape[1]):
            dem_extent, dem_bounds = ct.get_raster_extent(ds)
            mask_crop = gdal.Warp(os.path.join(direc,'mask_clip.tif'), dsm, outputBounds = tuple(dem_bounds))
            maskarr = mask_crop.GetRasterBand(1).ReadAsArray()
        elif (dem.shape[0] > maskarr.shape[0]) or (dem.shape[1] > maskarr.shape[1]):
            print(dem.shape[0],dem.shape[1])
            print(maskarr.shape[0],maskarr.shape[1])
            raise RuntimeError('The mask will not fully cover the DEM')
            
        if (dem.shape != maskarr.shape):
            print('dem is '+ str(dem.shape) +' and mask is '+ str(maskarr.shape))
            raise RuntimeError('Mask cannot be applied because DEM and mask are not the same shape')
        
        # apply mask
        masked_dem = dem*maskarr
        masked_dem[masked_dem==0]=nullval
    
        # export masked dem
        out_file = os.path.join(direc,'masked_dem'+str(dst_resoln)+'m.tif')
        ct.write_geotiff(out_file, masked_dem, ds, nullval)
        
    else:
        # check current DEM projection
        crsdem = osr.SpatialReference()
        crsdem.ImportFromWkt(ds.GetProjection())
        
        # reproject DEM, if needed
        if (crsdem.IsSame(dstcrs) == 0):
            print('Reprojecting DEM to ' + dst_crs)
            out_file = os.path.join(direc,'dem_reproject_'+str(dst_resoln)+'m.tif')
            ds = gdal.Warp(out_file, ds, dstSRS = dst_crs, xRes = dst_resoln, yRes = dst_resoln, dstNodata = nullval, resampleAlg = 'bilinear')
            dem = ds.GetRasterBand(1).ReadAsArray()
               
    
    if plot is True:
        if mask == True:
            # plot up DEM
            plt.figure()
            mskdem = ma.masked_where(masked_dem==nullval, masked_dem)
            plt.imshow(mskdem)
            plt.colorbar(label='Elevation (m)')
            plt.clim(vmin=0,vmax=2000)
            plt.show()
        else:
            # plot up DEM
            plt.figure()
            dem = ma.masked_where(dem==nullval, dem)
            plt.imshow(dem)
            plt.colorbar(label='Elevation (m)')
            plt.clim(vmin=0,vmax=2000)
            plt.show()
    
    # close out GDAL dataset objects
    ds = dsm = None
    
    return out_file


def classify_dem(dem_file, nullval, scaletype='quad', multiscale=False, wsize=7, coarsen=10, plot=True):
    """
    This function takes a masked dem and classifies it into landform types
    based on Wood (1996). There are several options available for scaling. The
    function exports a single landform raster (.tif) for single scale analysis
    (multiscale = False) and an additional raster for the multiscale analysis
    that provides a rough estimate of uncertainty.
    
    Wood, J. (1996). The geomorphological characterisation of digital
    elevation models. University of Leicester (United Kingdom).

    Parameters
    ----------
    dem_file : full file path to DEM geotiff file (string)
    nullval : value to input for NoData pixels (float)
    scaletype : option to select the scaling framework (default = 'quad'):
                'quad': local quadratic regression per window, terrain parameters
                        are derived directly from the coefficents according to Wood (1996)
                'agg': DEM is coarsened via mean aggregation first, terrain
                        parameters are derived with WhiteboxTools from coarsened DEM
                'gauss': terrain parameters are derived using gaussian space
                        scaling with WhiteboxTools (Newman et al, 2022)             
    multiscale : this option can only be selected with 'quad' scaletype for now
                    Performs multiscale analysis with varying window sizes (default = False)
    wsize : window size, for multiscale = False this is an integer
                        for multiscale = True, wsize is a range (e.g. wsize = range(3,25,2))
                        window sizes must be odd and step sizes must be even
    coarsen : this option is only used with 'agg' scaletype, amount of times to
                coarsen the DEM (e.g. 2 would result in half of the number of rows and columns)
                (int)

    """
    
    # dem directory
    direc = os.path.dirname(dem_file)
    
    
    # Mean aggregate scaling with Wood 1996 classification
    if scaletype == 'agg':
    
        # load dem as GDAL dataset object and array
        dem, ds = ct.read_geotiff(dem_file)
        
        _, resoln, _, _, _, _ = ds.GetGeoTransform()
        
        Tslope = 1.0 #np.median(slope)
        Tcurve = 1e-4 #np.median(plancurv)
        print('slope tolerance = '+str(Tslope))
        print('curvature tolerance ='+str(Tcurve))
        
        slope, plancurv, mincurv, maxcurv, dem_coarse = ct.generate_terrainparams(dem_file,coarsen,plot_terrain=False)
        
        featrs = ct.landform_class(slope,plancurv,mincurv,maxcurv,Tslope,Tcurve)
    
        
        # mask new feature array and save
        if isinstance(coarsen,int):
            m = np.array(dem_coarse==nullval,dtype=float)
            X = np.ones((coarsen,coarsen))
            new_mask = signal.correlate(m,X,'same')/(coarsen*coarsen)
            new_mask[np.where(new_mask < 1E-4)] = 0
            new_mask[np.where(new_mask > 0)] = 1
            featrs[np.where(new_mask==1.)]=nullval
        else:
            featrs[np.where(dem_coarse==nullval)]=nullval
        
        if plot is True:
            ct.plot_landforms(featrs, nullval)
            plt.title('Resolution = '+str(dst_resoln)+'m, Window Size(s) = '+str(wsize))
        
        # export landform rasters
        ct.write_geotiff(os.path.join(direc,'landforms_'+str(dst_resoln)+'m_'+str(coarsen)+'_agg.tif'),featrs,ds,nullval)
    
    
    # Gaussian Space Scale in Whitebox - Needs work, function does not output maximum or minimum curvature
    if scaletype == 'gauss':
        
        outfile = os.path.join(direc,'gauss_results','gauss_test.tif')
        outzfile = os.path.join(direc,'gauss_results','zscore.tif')
        outscfile = os.path.join(direc,'gauss_results','scale.tif')
        
        # Load whitebox tools and set working directory
        wbt = whitebox.WhiteboxTools()
        wbt.set_working_dir(direc)
        wbt.set_verbose_mode(False)  # if you want to print everything out, switch to True
        
        wbt.gaussian_scale_space(dem_file, outfile, outzfile, outscfile, points=None, 
            sigma=0, step=10, num_steps=5, lsp="Slope", z_factor=None)
    
    
    # Local Quadratic Regression
    if scaletype == 'quad' and multiscale is False:
        
        if not isinstance(wsize,int):
            raise RuntimeError('Multiple window sizes provided, select one for single scale analysis or \
                               set multiscale = True for multi-scale analysis')
                               
        if wsize % 2 == 0:
            raise RuntimeError('Window size must be odd')
        
        
        # read in file and pull cell resolution
        dem, ds = ct.read_geotiff(dem_file)
        _, resoln, _, _, _, _ = ds.GetGeoTransform()
        
        print('Finding landforms over ' + str(resoln*wsize) + ' m square window')
        
        # set slope and curvature thresholds
        Tslope = 1.0 #np.median(slope)
        Tcurve = 1e-4 #np.median(plancurv)
        print('slope tolerance = '+str(Tslope))
        print('curvature tolerance ='+str(Tcurve))
        
        start = datetime.now()
        
        # fit local quadratic function to window and get coefficients
        coeffs = ct.fit_local_quad(dem,resoln,resoln,wsize)
        a,b,c,d,e,f = coeffs
        
        slope = np.degrees(np.arctan(np.sqrt(d**2+e**2)))  # slope
        
        maxcurv = -a-b+np.sqrt((a-b)**2+c**2)  # maximal curvature
        mincurv = -a-b-np.sqrt((a-b)**2+c**2)  # minimal curvature
        
        crosscurv = -2*(b*d**2+a*e**2-c*d*e)/(d**2+e**2)  # cross-sectional curvature
        
        featrs = ct.landform_class(slope,crosscurv,mincurv,maxcurv,Tslope,Tcurve)  # get landform types based on Wood 1996
        
        # mask new feature array and save
        m = np.array(dem==nullval,dtype=float)
        X = np.ones((wsize,wsize))
        new_mask = signal.correlate(m,X,'same')/(wsize*wsize)
        new_mask[np.where(new_mask < 1E-4)] = 0
        new_mask[np.where(new_mask > 0)] = 1
        featrs[np.where(new_mask==1.)]=nullval
        
        
        # export landform rasters
        outfile = os.path.join(direc,'landforms_'+str(dst_resoln)+'m_'+str(wsize)+'_quad.tif')
        ct.write_geotiff(os.path.join(direc,'temp.tif'),featrs,ds,nullval)
        ds = None
        
        # clip edges of raster, moving window outside of raster results in unreliable classifications
        featrs, ds = ct.read_geotiff(os.path.join(direc,'temp.tif'))
        _,bounds = ct.get_raster_extent(ds)
        inc = [dst_resoln*int(wsize/2), dst_resoln*int(wsize/2), dst_resoln*-int(wsize/2), dst_resoln*-int(wsize/2)]
        dem_bounds = [sum(i) for i in zip(bounds,inc)]
        ds = gdal.Warp(outfile, ds, outputBounds = tuple(dem_bounds), dstNodata = nullval)
        
        # remove temp file
        ds = None
        os.remove(os.path.join(direc,'temp.tif'))
        
        end = datetime.now()
        td = (end - start).total_seconds()
        print('Done, time elapsed (minutes): ', td/60)
        
        # plot results
        if plot is True:
            print('Plotting up results')
            # read back in for plotting
            featrs,_ = ct.read_geotiff(outfile)
            ct.plot_landforms(featrs, nullval)
            plt.title('Resolution = '+str(dst_resoln)+'m, Window Size(s) = '+str(wsize))
        
            # plot up DEM
            plt.figure()
            mskdem = ma.masked_where(dem==nullval, dem)
            plt.imshow(mskdem)
            plt.colorbar(label='Elevation (m)')
    
    
    # multi-scale analysis  
    if scaletype == 'quad' and multiscale is True:
        
        if not isinstance(wsize,range):
            raise RuntimeError('wsize variable must be a range of window sizes, \
                               window sizes must be odd and step sizes must be even')
        
        if wsize.start % 2 == 0 or wsize.stop % 2 == 0:
            raise RuntimeError('Window sizes must be odd')
        
        if wsize.step % 2:
            raise RuntimeError('Step size must be even')
        
        # load dem as GDAL dataset object and array
        dem, ds = ct.read_geotiff(dem_file)
        _, resoln, _, _, _, _ = ds.GetGeoTransform()
        
        # set slope and curvature thresholds
        Tslope = 1.0 #np.median(slope)
        Tcurve = 1e-4 #np.median(plancurv)
        print('slope tolerance = '+str(Tslope))
        print('curvature tolerance ='+str(Tcurve))
        
        # allocate empty array for feature layer output
        lf = np.ones((dem.shape[0],dem.shape[1],len(wsize)))*nullval
        
        
        # main loop for multi-scale analysis
        start = datetime.now()
    
        for idx, win in enumerate(wsize):
            # fit local quadratic function to window and get coefficients
            coeffs = ct.fit_local_quad(dem,resoln,resoln,win)
            a,b,c,d,e,f = coeffs
            
            slope = np.degrees(np.arctan(np.sqrt(d**2+e**2)))  # slope
            
            maxcurv = -a-b+np.sqrt((a-b)**2+c**2)  # maximal curvature
            mincurv = -a-b-np.sqrt((a-b)**2+c**2)  # minimal curvature
            
            crosscurv = -2*(b*d**2+a*e**2-c*d*e)/(d**2+e**2)  # cross-sectional curvature
            
            featrs = ct.landform_class(slope,crosscurv,mincurv,maxcurv,Tslope,Tcurve)  # get landform types based on Wood 1996
            
            # np.save(os.path.join(direc,'landforms_'+str(win)+'.npy'),featrs)
            lf[:,:,idx] = featrs
            print('Window size '+ str(win) + ' - done')
        
        print('Finished local quad regression for all window sizes, combining...')    
        
        # aggregated landform classification via mode of all scales
        landform_all, landform_count = stats.mode(lf,2,keepdims=True)
        landform_all = landform_all[:,:,0]
        
        # mask new feature array and save
        m = np.array(dem==nullval,dtype=float)
        X = np.ones((wsize.stop,wsize.stop))
        new_mask = signal.correlate(m,X,'same')/(wsize.stop*wsize.stop)
        new_mask[np.where(new_mask < 1E-4)] = 0
        new_mask[np.where(new_mask > 0)] = 1
        # featrs = ma.masked_where(new_mask == 1.,featrs)
        landform_all[np.where(new_mask==1.)]=nullval
        
        
        # entropy
        probs = np.stack([np.sum(lf==c,axis=2) for c in range(6)],2).astype('float')
        probs /= np.tile(np.sum(probs,axis=2).reshape(probs.shape[0],probs.shape[1],1)+1e-8, (1,1,6))
        neg_entropy = np.sum(probs * np.log(probs+1e-7),axis=2)
        
        end = datetime.now()
        td = (end - start).total_seconds()
        print('Done, time elapsed (minutes): ', td/60)
        print('Writing out files...')
        
        # export mode and entropy to tif files
        mode_outfile = 'landforms_'+str(dst_resoln)+'m_'+str(wsize)+'_mode.tif'
        entropy_outfile = 'landforms_'+str(dst_resoln)+'m_'+str(wsize)+'_entropy.tif'
        ct.write_geotiff(os.path.join(direc,mode_outfile),landform_all,ds, nullval)
        ct.write_geotiff(os.path.join(direc,entropy_outfile),neg_entropy,ds, nullval)
        
        print('Done')
        
        # plot up results
        if plot is True:
            ct.plot_landforms(landform_all, nullval)
            plt.title('Resolution = '+str(dst_resoln)+'m, Window Size(s) = '+str(wsize))
            
            plt.figure()
            plt.imshow(neg_entropy)
            plt.colorbar()
            plt.show()
        
    ds = None
    

if __name__=='__main__':
    
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    preprocess = config.get('Run Preprocessor',True)
        
    raw_file = config['Files']['DEM']
    mask_file = config['Files']['Mask']
    
    nullval = config.get('NoData Value', -32767.0)
    dst_epsg = config['Target EPSG']
    dst_resoln = config['Target Resolution']
    maskopt = config.get('Use Mask', False)
    plot1 = config.get('Generate Initial DEM Plot', True)
    
    scaletype = config.get('Scaling Type','quad')
    multiscale = config.get('Multiscale Analysis', False)
    if multiscale == False:
        wsize = config['Window Size']
    else:
        wsize = range(config['Window Size'][0],config['Window Size'][1],config['Window Size'][2])
    coarsen = config.get('Coarsen', None)
    plot2 = config.get('Generate Classified DEM Plot', True)
    
    if preprocess == True:
        DEM_file = preprocess_dem(raw_file, mask_file, nullval, dst_epsg, dst_resoln, mask=maskopt, plot=plot1)
        classify_dem(DEM_file, nullval, scaletype=scaletype, multiscale=multiscale, wsize=wsize, coarsen=coarsen, plot=plot2)
    
    else:
        if maskopt == True:
            DEM_file = os.path.join(os.path.dirname(raw_file),'masked_dem'+str(dst_resoln)+'m.tif')
        else:
            DEM_file = os.path.join(os.path.dirname(raw_file),'dem_reproject_'+str(dst_resoln)+'m.tif')
            
        classify_dem(DEM_file, nullval, scaletype=scaletype, multiscale=multiscale, wsize=wsize, coarsen=coarsen, plot=plot2)
