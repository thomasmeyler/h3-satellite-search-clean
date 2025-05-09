#!/usr/bin/env python3
"""
Combine all port tiles into a single GeoJSON file.
"""

import os
import geopandas as gpd
from utils.ports import get_all_ports

def combine_tiles(output_path):
    """
    Combine all port tiles into a single GeoJSON file.
    
    Args:
        output_path: Path to save the combined GeoJSON file
        
    Returns:
        GeoDataFrame with all tiles
    """
    # Set up directories
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    processed_dir = os.path.join(data_dir, "processed")
    
    # Get all ports
    ports = get_all_ports()
    
    # Load tiles for each port
    all_tiles = []
    
    for port in ports:
        port_tiles_path = os.path.join(processed_dir, f"{port}_tiles.geojson")
        
        if os.path.exists(port_tiles_path):
            port_tiles = gpd.read_file(port_tiles_path)
            all_tiles.append(port_tiles)
            print(f"Loaded {len(port_tiles)} tiles for {port}")
        else:
            print(f"No tiles found for {port}")
    
    # Combine all tiles
    if all_tiles:
        combined_tiles = gpd.GeoDataFrame(pd.concat(all_tiles, ignore_index=True))
        combined_tiles.to_file(output_path, driver="GeoJSON")
        print(f"Saved {len(combined_tiles)} tiles to {output_path}")
        return combined_tiles
    else:
        print("No tiles found")
        return None

if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Combine all port tiles into a single GeoJSON file")
    parser.add_argument("--output", type=str, default="data/processed/all_tiles.geojson", help="Path to save the combined GeoJSON file")
    
    args = parser.parse_args()
    
    # Combine tiles
    combine_tiles(args.output)