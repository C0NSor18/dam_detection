# Dam Detection using Neural Networks

Welcome to my blog on dam detection using neural networks. This projects primarily aims to detect dams using satellite data!

## Table of Contents
[Exploring Earth Engine](#Working with Earth Engine)


*1 August 2019*
## Exploring Earth Engine
 
Gathering data to work with has been sort of a challenge. Since I had no prior experience with javascript and Google's documentation on its Earth Engine API is rather minimal, I was mostly left with the tutorials and whatever I could find on the [gis stackexchange](https://gis.stackexchange.com). 

So the basic idea is that I found several datasets with manually annotated dam locations in shapefile format (WGS84) from the [Global Dam Watch](http://globaldamwatch.org/). The dataset I am using for now is called GRanD, which contains over 7000 locations of very large dams. The next logical step is to import this shapefile into Earth Engine. This was rather easy, as Google Earth Engine allows you to import external files as ```assets```, and import them as featurecollections in the code editor. Since I am completely new to GIS software the easiest thing to do is to visualize the dam locations, which looks like this:
![](images/grand_dams.png)

Eyeballing the locations the coordinates seem to be in the right places, altough a little far off from the actual dam locations at times, but we'll just have to live with that. The next step is that I want to extract and download image patches around these coordinates so that I can feed them into a classifier for training. This turned out to be more difficult than I wanted, as the Earth Engine docs did not explicitly cover this. Luckily I found a blog post by [Charlotte Weil](https://medium.com/@charlotteweil/can-we-locate-dams-from-space-2a796ac8c04b) covering the same topic. In order to get to the result that we want we have take several steps, explained below.

The first step is to choose a satellite and the layers you want to use. For this project, I used the [Sentinel 2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2) Multispectral Instrument (MSI) with Level-1C orthorectified top-of-atmosphere reflectance. The most common problem with using satellite images is that they can be obstructed by clouds. Luckily GEE provides a code that removes most of the clouds right off the bat:

```javscript
/**
 * Function to mask clouds using the Sentinel-2 QA band
 * @param {ee.Image} image Sentinel-2 image
 * @return {ee.Image} cloud masked Sentinel-2 image
 */
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

// Map the function over one year of data and take the median.
// Load Sentinel-2 TOA reflectance data.
var dataset = ee.ImageCollection('COPERNICUS/S2')
                  .filterDate('2018-01-01', '2018-06-30')
                  // Pre-filter to get less cloudy granules.
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .map(maskS2clouds);
```

We will extract the RGB bands, which are encoded as B4, B3, and B2 in S2, and compute the Normalized Difference Water Index (NDWI), for which we also need the Near Infrared (NIR) band, which is named B8 in S2. All of the bands used from the S2 satellite are sampled at 10m resolution. Following the blog post from Charlotte, elevation is also factored in by using the Alos DSM. This brings up to a total of 5 bands (channels) to use for training.

