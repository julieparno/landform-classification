# Preprocess NSIDC-0780 Ice Masks
import numpy as np
import os
from osgeo import gdal
import classify_tools as ct

direc = '..\\..\\Data\\Ice_Mask\\NSIDC-0780'
filename = 'NSIDC-0780_SeaIceRegions_PS-N3.125km_v1.0.nc'
out_file = 'ice_mask.tif'
nullval = -32767.0

file = os.path.join(direc,filename)

var_name = 'sea_ice_region_surface_mask'
sublayername =  gdal.Open('NETCDF:"{0}":{1}'.format(file, var_name ),gdal.GA_ReadOnly)
ds = gdal.Translate(os.path.join(direc,'temp.tif'),sublayername,outputSRS = 'EPSG:3413', format='GTiff')

ds = None

# load mask as GDAL dataset object and array
maskarr, dsm = ct.read_geotiff(os.path.join(direc,'temp.tif'))

maskarr[maskarr == 1] = nullval
maskarr[maskarr == 33] = 1
maskarr[maskarr == 34] = 1
maskarr[maskarr != 1] = 0

ct.write_geotiff(os.path.join(direc,out_file), maskarr, dsm, nullval)

dsm = None