# landform-classification

Workflow for classifying landforms from digital elevation models.  Classification criteria are based on those from Wood (1996).

## Installation

1. Create conda environment and install dependencies
```
conda env create -f DEM_class.yml
```
2. Create a config.yml file with the following information

```yaml
# Configuration file for process_DEM.py

# WhiteBox requires full path for DEM
Files:
  DEM: "C:\\Users\\JohnSmith\\Documents\\Projects\\LandClass\\raw_dem.tif"    # full path to raw DEM file
  Mask: "..\\..\\Data\\Ice_Mask\\NSIDC-0714\\GimpIceMask_90m_2015_v1.2.tif"   # relative path to ice mask

Run Preprocessor: False

NoData Value: -32767.0    # NoData value
Target EPSG: 3413         # generally EPSG:3413 for Arctic, EPSG:3976 for Antarctic
Target Resolution: 30     # in units of target projection

Use Mask: False
Generate Initial DEM Plot: False

Scaling Type: "quad"          # "quad" (default), "agg", or "gauss"
Multiscale Analysis: True
Window Size: [9,55,4]         # single integer for multiscale=False, list of range values for multiscale=True, ex. [start, stop, step]
Coarsen: None                 # only used for "agg" scaling type
Generate Classified DEM Plot: True

```
3. Run process_DEM.py


## References
Wood, Joseph. 1996. “The Geomorphological Characterisation of Digital Elevation Models.” Ph.D., England: University of Leicester (United Kingdom). https://www.proquest.com/docview/301562226/abstract/7B0AB0ABA5A4EDBPQ/1. 
