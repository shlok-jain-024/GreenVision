// GreenGrid plantable-land export — paste into https://code.earthengine.google.com/
//
// NDVI says where vegetation is ABSENT, but not whether the land is available
// (bare plot) or occupied (concrete, rooftops). This script uses Google's
// Dynamic World land-cover classification (10 m) to estimate, for each grid
// point, the fraction of surrounding land that is BARE GROUND — i.e. actually
// plantable. Exports one CSV to Drive: lon, lat, mean  (mean = 0..1 bare
// probability averaged over the past year and a 100 m neighborhood).
//
// This is a FREE ALTERNATIVE to the Esri 10 m Land Cover route in the README.
// It produces a per-point 0..1 fraction, so it feeds the LEGACY plantable slot
// (not the richer per-patch site finder). In GreenGrid's config/city.yaml:
//   sites:
//     plantable_csv: data/ahmedabad_plantable.csv
// For point-level planting_sites.geojson output, export bare-ground CENTROIDS
// as lat,lon,patch_area_m2 instead and use sites.candidates_csv.

var BBOX = [72.45, 22.90, 72.70, 23.15]; // lon_min, lat_min, lon_max, lat_max
var GRID_STEP = 0.01;                    // match your NDVI export grid
var END = ee.Date(Date.now());
var START = END.advance(-12, 'month');   // average over the last year

var bbox = ee.Geometry.Rectangle(BBOX);

var points = [];
for (var lon = BBOX[0]; lon < BBOX[2]; lon += GRID_STEP) {
  for (var lat = BBOX[1]; lat < BBOX[3]; lat += GRID_STEP) {
    points.push(ee.Feature(ee.Geometry.Point([lon, lat]), {
      lon: Math.round(lon * 10000) / 10000,
      lat: Math.round(lat * 10000) / 10000
    }));
  }
}
var grid = ee.FeatureCollection(points);

// Mean probability of the "bare" class over the period. If you also want to
// count sparse grassland as plantable, add .select(['bare','grass']) and sum.
var bare = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(bbox)
  .filterDate(START, END)
  .select('bare')
  .mean();

var sampled = bare.reduceRegions({
  collection: grid,
  reducer: ee.Reducer.mean(),
  scale: 100
});

Export.table.toDrive({
  collection: sampled,
  description: 'ahmedabad_plantable',
  fileFormat: 'CSV',
  selectors: ['lon', 'lat', 'mean']
});
// Run, start the task in the Tasks tab, download the CSV into data/.
