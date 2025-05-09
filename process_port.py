#!/usr/bin/env python3
"""
Process a single port through the entire pipeline.

This script runs the complete pipeline for a single port:
1. Generate H3 tiles
2. Download satellite imagery
3. Generate embeddings
4. Index embeddings in vector database
"""

import os
import argparse
import geopandas as gpd
import ee
from utils.ports import get_all_ports
from utils.tiling import generate_port_tiles, DEFAULT_H3_RESOLUTION
from utils.imagery import authenticate_gee, download_imagery_for_port
from utils.embeddings import generate_embeddings_for_port
from utils.indexing import index_embeddings, connect_to_qdrant

def process_port(port_name, resolution=DEFAULT_H3_RESOLUTION, visualize=False, 
                 start_date=None, end_date=None, max_cloud_cover=None, recreate_index=False):
    """
    Process a single port through the entire pipeline.
    
    Args:
        port_name: Name of the port
        resolution: H3 resolution
        visualize: Whether to visualize the results
        start_date: Start date for imagery (YYYY-MM-DD)
        end_date: End date for imagery (YYYY-MM-DD)
        max_cloud_cover: Maximum cloud cover percentage
        recreate_index: Whether to recreate the vector index
        
    Returns:
        None
    """
    print(f"\n{'='*50}")
    print(f"Processing port: {port_name}")
    print(f"{'='*50}\n")
    
    # Set up directories
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    processed_dir = os.path.join(data_dir, "processed")
    raw_dir = os.path.join(data_dir, "raw")
    embeddings_dir = os.path.join(data_dir, "embeddings")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Step 1: Generate H3 tiles
    print(f"\nStep 1: Generating H3 tiles for {port_name} at resolution {resolution}")
    tiles_path = os.path.join(processed_dir, f"{port_name}_tiles.geojson")
    tiles_gdf = generate_port_tiles(port_name, resolution, tiles_path, visualize)
    
    if len(tiles_gdf) == 0:
        print(f"No tiles generated for {port_name}. Exiting.")
        return
    
    # Step 2: Download satellite imagery
    print(f"\nStep 2: Downloading satellite imagery for {port_name}")
    
    # Authenticate with Google Earth Engine
    credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gee_credentials.json")
    authenticate_gee(credentials_path)
    
    # Download imagery
    from utils.imagery import DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_MAX_CLOUD_COVER
    
    # Use default values if not provided
    if start_date is None:
        start_date = DEFAULT_START_DATE
    if end_date is None:
        end_date = DEFAULT_END_DATE
    if max_cloud_cover is None:
        max_cloud_cover = DEFAULT_MAX_CLOUD_COVER
    
    image_paths = download_imagery_for_port(
        port_name, tiles_gdf, raw_dir, 
        start_date, end_date, max_cloud_cover
    )
    
    if len(image_paths) == 0:
        print(f"No imagery downloaded for {port_name}. Exiting.")
        return
    
    # Step 3: Generate embeddings
    print(f"\nStep 3: Generating embeddings for {port_name}")
    embedding_paths = generate_embeddings_for_port(port_name, raw_dir, embeddings_dir)
    
    if len(embedding_paths) == 0:
        print(f"No embeddings generated for {port_name}. Exiting.")
        return
    
    # Step 4: Index embeddings
    print(f"\nStep 4: Indexing embeddings for {port_name}")
    
    # Connect to Qdrant
    vector_db_path = os.path.join(processed_dir, "vector_db")
    client = connect_to_qdrant(vector_db_path)
    
    # Index embeddings
    index_embeddings(client, port_name, embeddings_dir, recreate=recreate_index)
    
    print(f"\n{'='*50}")
    print(f"Completed processing for port: {port_name}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single port through the entire pipeline")
    parser.add_argument("--port", type=str, required=True, help="Port name (tianjin, dalian, shanghai, qingdao)")
    parser.add_argument("--resolution", type=int, default=DEFAULT_H3_RESOLUTION, help=f"H3 resolution (default: {DEFAULT_H3_RESOLUTION})")
    parser.add_argument("--visualize", action="store_true", help="Visualize the results")
    parser.add_argument("--start_date", type=str, help="Start date for imagery (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="End date for imagery (YYYY-MM-DD)")
    parser.add_argument("--max_cloud_cover", type=float, help="Maximum cloud cover percentage")
    parser.add_argument("--recreate_index", action="store_true", help="Recreate the vector index")
    
    args = parser.parse_args()
    
    # Validate port name
    valid_ports = get_all_ports()
    if args.port.lower() not in valid_ports:
        print(f"Error: Invalid port name. Valid options are: {', '.join(valid_ports)}")
        exit(1)
    
    # Process the port
    process_port(
        args.port.lower(),
        args.resolution,
        args.visualize,
        args.start_date,
        args.end_date,
        args.max_cloud_cover,
        args.recreate_index
    )