// GreenGrid NDVI export — paste this into https://code.earthengine.google.com/
// (free Google Earth Engine account required: https://earthengine.google.com/signup/)
//
// What it does: for every month since START, builds a cloud-filtered monthly
// Sentinel-2 composite over the Ahmedabad bbox, computes NDVI, samples it on
// a ~1.1 km grid of points, and exports one CSV to your Google Drive with
// columns: lon, lat, month, mean   (month 0 = the START month, oldest first).
//
// Then in GreenGrid's config/city.yaml:
//   data.start_month: 1            # January, matching START below
//   adapters.green_cover: "csv:data/ahmedabad_ndvi_monthly.csv"
//
// Keep START's calendar month aligned with data.start_month, and use the
// same START for your traffic and AQI series so month indices line up.

var BBOX = [72.45, 22.90, 72.70, 23.15]; // lon_min, lat_min, lon_max, lat_max
var START = '2016-01-01';                // month index 0
var N_MONTHS = 126;                      // through mid-2026; adjust as needed
var GRID_STEP = 0.01;                    // ~1.1 km between sample points

var bbox = ee.Geometry.Rectangle(BBOX);

// Build the sampling grid.
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

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(bbox)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30));

function monthlyNdvi(m) {
  var start = ee.Date(START).advance(m, 'month');
  var end = start.advance(1, 'month');
  var composite = s2.filterDate(start, end).median();
  var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('ndvi');
  // mean NDVI in a 100 m neighborhood around each grid point;
  // months with no cloud-free imagery yield empty values (dropped later)
  return ndvi.reduceRegions({
    collection: grid,
    reducer: ee.Reducer.mean(),
    scale: 100
  }).map(function (f) { return f.set('month', m); });
}

var all = ee.FeatureCollection(
  ee.List.sequence(0, N_MONTHS - 1).map(monthlyNdvi)
).flatten();

Export.table.toDrive({
  collection: all,
  description: 'ahmedabad_ndvi_monthly',
  fileFormat: 'CSV',
  selectors: ['lon', 'lat', 'month', 'mean']
});
// Click "Run", then start the task in the Tasks tab (may take ~tens of
// minutes). Download the CSV from Drive into this project's data/ folder.
