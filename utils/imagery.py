"""
Satellite imagery utilities for downloading and processing imagery.

This module provides functions for downloading satellite imagery from Google Earth Engine.
"""

import os
import json
import tempfile
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from PIL import Image
import ee
from tqdm import tqdm

from utils.ports import get_port_bbox, get_all_ports

# Default imagery parameters
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_MAX_CLOUD_COVER = 20  # Maximum cloud cover percentage
DEFAULT_SCALE = 10  # 10m resolution for Sentinel-2

def authenticate_gee(service_account_path):
    """
    Authenticate with Google Earth Engine.
    
    Args:
        service_account_path: Path to service account credentials JSON file
        
    Returns:
        None
    """
    try:
        # Use service account
        with open(service_account_path, 'r') as f:
            creds = json.load(f)
        credentials = ee.ServiceAccountCredentials(
            creds['client_email'], 
            service_account_path
        )
        ee.Initialize(credentials)
        print("Successfully authenticated with Google Earth Engine")
    except Exception as e:
        print(f"Error authenticating with GEE: {e}")
        raise

def create_roi_geometry(bbox):
    """
    Create an Earth Engine geometry from a bounding box.
    
    Args:
        bbox: Bounding box [min_lng, min_lat, max_lng, max_lat]
        
    Returns:
        ee.Geometry
    """
    return ee.Geometry.Rectangle(bbox)

def get_sentinel2_collection(roi, start_date, end_date, max_cloud_cover=DEFAULT_MAX_CLOUD_COVER):
    """
    Get a collection of Sentinel-2 images for a region of interest.
    
    Args:
        roi: Earth Engine geometry
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_cloud_cover: Maximum cloud cover percentage
        
    Returns:
        ee.ImageCollection
    """
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover)))

def get_least_cloudy_image(collection):
    """
    Get the least cloudy image from a collection.
    
    Args:
        collection: Earth Engine image collection
        
    Returns:
        ee.Image
    """
    return collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()

def download_image_for_tile(image, tile_polygon, output_path, bands=['B4', 'B3', 'B2'], scale=DEFAULT_SCALE):
    """
    Download an image for a specific tile.
    
    Args:
        image: Earth Engine image
        tile_polygon: Shapely polygon representing the tile
        output_path: Path to save the image
        bands: List of bands to download
        scale: Resolution in meters
        
    Returns:
        Path to downloaded image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert tile to ee.Geometry
    coords = list(tile_polygon.exterior.coords)
    ee_polygon = ee.Geometry.Polygon(coords)
    
    # For H3 resolution 6 tiles, we can use a better resolution
    # H3 resolution 6 tiles are smaller than resolution 4
    adjusted_scale = scale * 2  # Use 20m resolution instead of 10m
    
    try:
        # Get the image data
        url = image.visualize(bands=bands, min=0, max=3000).getDownloadURL({
            'region': ee_polygon,
            'scale': adjusted_scale,
            'format': 'GeoTIFF'
        })
        
        # Download the image
        with tempfile.NamedTemporaryFile(suffix='.tif') as tmp:
            os.system(f"curl -s '{url}' > {tmp.name}")
            # Copy to output path
            os.system(f"cp {tmp.name} {output_path}")
        
        return output_path
    except Exception as e:
        print(f"Error downloading image at scale {adjusted_scale}m: {e}")
        
        # Try with an even higher scale if the first attempt failed
        try:
            higher_scale = adjusted_scale * 2
            print(f"Retrying with scale {higher_scale}m...")
            
            url = image.visualize(bands=bands, min=0, max=3000).getDownloadURL({
                'region': ee_polygon,
                'scale': higher_scale,
                'format': 'GeoTIFF'
            })
            
            with tempfile.NamedTemporaryFile(suffix='.tif') as tmp:
                os.system(f"curl -s '{url}' > {tmp.name}")
                os.system(f"cp {tmp.name} {output_path}")
            
            return output_path
        except Exception as e2:
            print(f"Error downloading image at scale {higher_scale}m: {e2}")
            raise

def download_imagery_for_port(port_name, tiles_gdf, output_dir, 
                             start_date=DEFAULT_START_DATE, 
                             end_date=DEFAULT_END_DATE, 
                             max_cloud_cover=DEFAULT_MAX_CLOUD_COVER,
                             scale=DEFAULT_SCALE):
    """
    Download imagery for all tiles in a port.
    
    Args:
        port_name: Name of the port
        tiles_gdf: GeoDataFrame with H3 tiles
        output_dir: Directory to save the imagery
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_cloud_cover: Maximum cloud cover percentage
        scale: Resolution in meters
        
    Returns:
        List of paths to downloaded images
    """
    # Filter tiles for the specified port
    port_tiles = tiles_gdf[tiles_gdf['port'] == port_name]
    
    if len(port_tiles) == 0:
        print(f"No tiles found for port: {port_name}")
        return []
    
    # Get port bounding box
    bbox = get_port_bbox(port_name)
    
    # Create ROI geometry
    roi_geometry = create_roi_geometry(bbox)
    
    # Get Sentinel-2 collection
    s2_collection = get_sentinel2_collection(roi_geometry, start_date, end_date, max_cloud_cover)
    image_count = s2_collection.size().getInfo()
    print(f"Found {image_count} Sentinel-2 images for {port_name}")
    
    if image_count == 0:
        print(f"No imagery found for {port_name}. Try adjusting date range or cloud cover threshold.")
        return []
    
    # Get the least cloudy image
    best_image = get_least_cloudy_image(s2_collection)
    image_date = ee.Date(best_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    cloud_cover = best_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    print(f"Selected image from {image_date} with {cloud_cover:.2f}% cloud cover")
    
    # Create output directory
    port_output_dir = os.path.join(output_dir, port_name)
    os.makedirs(port_output_dir, exist_ok=True)
    
    # Download imagery for each tile
    image_paths = []
    
    for idx, row in tqdm(port_tiles.iterrows(), total=len(port_tiles), desc=f"Downloading imagery for {port_name}"):
        h3_index = row['h3_index']
        tile_polygon = row['geometry']
        
        output_path = os.path.join(port_output_dir, f"{h3_index}.tif")
        
        try:
            download_image_for_tile(best_image, tile_polygon, output_path)
            image_paths.append(output_path)
        except Exception as e:
            print(f"Error downloading tile {h3_index}: {e}")
    
    print(f"Downloaded {len(image_paths)} images for {port_name}")
    return image_paths

def download_imagery_for_all_ports(tiles_gdf, output_dir, 
                                  start_date=DEFAULT_START_DATE, 
                                  end_date=DEFAULT_END_DATE, 
                                  max_cloud_cover=DEFAULT_MAX_CLOUD_COVER,
                                  scale=DEFAULT_SCALE):
    """
    Download imagery for all ports.
    
    Args:
        tiles_gdf: GeoDataFrame with H3 tiles
        output_dir: Directory to save the imagery
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_cloud_cover: Maximum cloud cover percentage
        scale: Resolution in meters
        
    Returns:
        Dictionary of image paths for each port
    """
    all_image_paths = {}
    
    for port_name in get_all_ports():
        image_paths = download_imagery_for_port(
            port_name, tiles_gdf, output_dir, 
            start_date, end_date, max_cloud_cover, scale
        )
        all_image_paths[port_name] = image_paths
    
    return all_image_paths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download satellite imagery for ports of interest")
    parser.add_argument("--port", type=str, help="Port name (tianjin, dalian, shanghai, qingdao, or 'all')")
    parser.add_argument("--tiles_path", type=str, default="../data/processed/all_tiles.geojson", help="Path to tiles GeoJSON file")
    parser.add_argument("--output_dir", type=str, default="../data/raw", help="Output directory for imagery")
    parser.add_argument("--start_date", type=str, default=DEFAULT_START_DATE, help=f"Start date (default: {DEFAULT_START_DATE})")
    parser.add_argument("--end_date", type=str, default=DEFAULT_END_DATE, help=f"End date (default: {DEFAULT_END_DATE})")
    parser.add_argument("--max_cloud_cover", type=float, default=DEFAULT_MAX_CLOUD_COVER, help=f"Maximum cloud cover percentage (default: {DEFAULT_MAX_CLOUD_COVER})")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE, help=f"Resolution in meters (default: {DEFAULT_SCALE})")
    parser.add_argument("--credentials", type=str, default="../gee_credentials.json", help="Path to GEE credentials JSON file")
    
    args = parser.parse_args()
    
    # Authenticate with Google Earth Engine
    authenticate_gee(args.credentials)
    
    # Load tiles
    tiles_gdf = gpd.read_file(args.tiles_path)
    
    if args.port and args.port.lower() != "all":
        # Download imagery for a specific port
        download_imagery_for_port(
            args.port, tiles_gdf, args.output_dir, 
            args.start_date, args.end_date, args.max_cloud_cover, args.scale
        )
    else:
        # Download imagery for all ports
        download_imagery_for_all_ports(
            tiles_gdf, args.output_dir, 
            args.start_date, args.end_date, args.max_cloud_cover, args.scale
        )