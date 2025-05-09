"""
H3 tiling utilities for satellite imagery analysis.

This module provides functions for generating H3 tiles for regions of interest.
"""

import os
import json
import h3
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point, box
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.ports import get_port_bbox, get_port_coordinates, get_all_ports

# Default H3 resolution
DEFAULT_H3_RESOLUTION = 6

def point_to_h3(lat, lng, resolution=DEFAULT_H3_RESOLUTION):
    """
    Convert a point to an H3 cell index.
    
    Args:
        lat: Latitude
        lng: Longitude
        resolution: H3 resolution (default: 4)
        
    Returns:
        H3 cell index
    """
    return h3.latlng_to_cell(lat, lng, resolution)

def h3_to_polygon(h3_index):
    """
    Convert an H3 cell index to a Shapely polygon.
    
    Args:
        h3_index: H3 cell index
        
    Returns:
        Shapely polygon
    """
    boundary = h3.cell_to_boundary(h3_index)
    # H3 returns [lat, lng] but Shapely expects [lng, lat]
    return Polygon([(point[1], point[0]) for point in boundary])

def bbox_to_h3_cells(bbox, resolution=DEFAULT_H3_RESOLUTION):
    """
    Convert a bounding box to a set of H3 cells.
    
    Args:
        bbox: Bounding box [min_lng, min_lat, max_lng, max_lat]
        resolution: H3 resolution (default: 4)
        
    Returns:
        Set of H3 cell indices
    """
    min_lng, min_lat, max_lng, max_lat = bbox
    
    # Create a polygon from the bounding box
    polygon = box(min_lng, min_lat, max_lng, max_lat)
    
    # Convert to list of coordinates expected by LatLngPoly
    # LatLngPoly expects [(lat, lng)] format
    coords = [(p[1], p[0]) for p in polygon.exterior.coords]
    
    # Create a LatLngPoly
    from h3 import LatLngPoly
    lat_lng_poly = LatLngPoly(coords)
    
    # Get H3 cells that intersect with the polygon
    h3_indices = h3.h3shape_to_cells(lat_lng_poly, resolution)
    
    return h3_indices

def generate_port_tiles(port_name, resolution=DEFAULT_H3_RESOLUTION, save_path=None, visualize=False):
    """
    Generate H3 tiles for a specific port.
    
    Args:
        port_name: Name of the port
        resolution: H3 resolution (default: 4)
        save_path: Path to save the tiles (default: None)
        visualize: Whether to visualize the tiles (default: False)
        
    Returns:
        GeoDataFrame with H3 tiles
    """
    # Get port bounding box
    bbox = get_port_bbox(port_name)
    
    # Get H3 cells
    h3_indices = bbox_to_h3_cells(bbox, resolution)
    print(f"Generated {len(h3_indices)} H3 cells for {port_name} at resolution {resolution}")
    
    # Convert to GeoDataFrame
    geometries = []
    for h3_index in h3_indices:
        polygon = h3_to_polygon(h3_index)
        geometries.append({
            'h3_index': h3_index,
            'port': port_name,
            'geometry': polygon
        })
    
    # Create GeoDataFrame
    if geometries:
        gdf = gpd.GeoDataFrame(geometries, crs="EPSG:4326")
    else:
        # If no geometries were created, return an empty GeoDataFrame with the correct structure
        gdf = gpd.GeoDataFrame(
            columns=['h3_index', 'port', 'geometry'],
            geometry='geometry',
            crs="EPSG:4326"
        )
    
    # Save to file if specified
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        gdf.to_file(save_path, driver="GeoJSON")
        print(f"Saved {len(gdf)} tiles to {save_path}")
    
    # Visualize if specified
    if visualize:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot port bounding box
        bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
        bbox_gdf = gpd.GeoDataFrame([{'geometry': bbox_poly}], crs="EPSG:4326")
        bbox_gdf.plot(ax=ax, color='none', edgecolor='red', linewidth=2)
        
        # Plot H3 cells if there are any
        if not gdf.empty:
            gdf.plot(ax=ax, color='none', edgecolor='blue', alpha=0.7)
        
        # Add port name
        port_coords = get_port_coordinates(port_name)
        ax.scatter(port_coords['lng'], port_coords['lat'], color='red', s=50)
        ax.text(port_coords['lng'], port_coords['lat'], port_name.upper(), 
                fontsize=12, ha='center', va='bottom', color='red')
        
        ax.set_title(f"{port_name.capitalize()} Port - H3 Resolution {resolution}")
        plt.tight_layout()
        
        # Save figure if save_path is specified
        if save_path:
            fig_path = os.path.join(os.path.dirname(save_path), f"{port_name}_tiles.png")
            plt.savefig(fig_path, dpi=300)
            print(f"Saved visualization to {fig_path}")
        
        plt.show()
    
    return gdf

def generate_all_port_tiles(resolution=DEFAULT_H3_RESOLUTION, output_dir=None, visualize=False):
    """
    Generate H3 tiles for all ports.
    
    Args:
        resolution: H3 resolution (default: 4)
        output_dir: Directory to save the tiles (default: None)
        visualize: Whether to visualize the tiles (default: False)
        
    Returns:
        Dictionary of GeoDataFrames with H3 tiles for each port
    """
    port_tiles = {}
    
    for port_name in get_all_ports():
        if output_dir:
            save_path = os.path.join(output_dir, f"{port_name}_tiles.geojson")
        else:
            save_path = None
        
        gdf = generate_port_tiles(port_name, resolution, save_path, visualize)
        port_tiles[port_name] = gdf
    
    # Combine all tiles into a single GeoDataFrame
    if output_dir:
        all_tiles = pd.concat(port_tiles.values())
        all_tiles.to_file(os.path.join(output_dir, "all_tiles.geojson"), driver="GeoJSON")
        print(f"Saved {len(all_tiles)} tiles for all ports to {os.path.join(output_dir, 'all_tiles.geojson')}")
    
    return port_tiles

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate H3 tiles for ports of interest")
    parser.add_argument("--port", type=str, help="Port name (tianjin, dalian, shanghai, qingdao, or 'all')")
    parser.add_argument("--resolution", type=int, default=DEFAULT_H3_RESOLUTION, help=f"H3 resolution (default: {DEFAULT_H3_RESOLUTION})")
    parser.add_argument("--output_dir", type=str, default="../data/processed", help="Output directory for tiles")
    parser.add_argument("--visualize", action="store_true", help="Visualize the tiles")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.port and args.port.lower() != "all":
        # Generate tiles for a specific port
        save_path = os.path.join(args.output_dir, f"{args.port.lower()}_tiles.geojson")
        generate_port_tiles(args.port, args.resolution, save_path, args.visualize)
    else:
        # Generate tiles for all ports
        generate_all_port_tiles(args.resolution, args.output_dir, args.visualize)