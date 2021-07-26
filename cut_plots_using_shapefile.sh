#!/bin/sh
''' 
By Monica Danilevicz

This script intends to cut the plots from the orthomosaic by using already designed shapefiles
the shapefiles and the original orthomosaic.tif must be in the same file.
Based on a thread https://gis.stackexchange.com/questions/118236/clip-raster-by-shapefile-in-parts

Ideally, run this scrip from directory that has the orthomosaic.tiff and a folder with the shapefiles

I have used GDAL from the docker container: osgeo/gdal:alpine-small-latest
'''

# Run this line to clip the images, first separate the polygon
singularity_image=$MYSCRATCH/singularity/gdal_alpine-small-latest.sif

for image in $(ls *.tif) ;
    # for all orthomosaic tif files in the folder
	do dir_name=$(basename ${image} .tif)
	mkdir ${dir_name}_test
	for plot in $(ls ${dir_name}/*shp) ;
		do fname=$(basename $plot .shp)
		singularity exec $singularity_image gdalwarp -srcnodata -32767 \
		-cutline $plot -crop_to_cutline -of GTiff  -dstnodata None \
		$image $dir_name'_test/'${fname}.tiff
		done
	done

