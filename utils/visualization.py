"""
Visualization utilities for satellite imagery and search results.

This module provides functions for visualizing satellite imagery and search results.
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from PIL import Image
from qdrant_client.http import models as qmodels

def visualize_port_tiles(port_name, tiles_gdf, output_path=None):
    """
    Visualize tiles for a specific port.
    
    Args:
        port_name: Name of the port
        tiles_gdf: GeoDataFrame with H3 tiles
        output_path: Path to save the visualization
        
    Returns:
        None
    """
    # Filter tiles for the specified port
    port_tiles = tiles_gdf[tiles_gdf['port'] == port_name]
    
    if len(port_tiles) == 0:
        print(f"No tiles found for port: {port_name}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot tiles
    port_tiles.plot(ax=ax, color='none', edgecolor='blue', alpha=0.7)
    
    # Add tile IDs
    for idx, row in port_tiles.iterrows():
        centroid = row['geometry'].centroid
        ax.text(centroid.x, centroid.y, row['h3_index'][-4:], 
                fontsize=8, ha='center', va='center', color='blue')
    
    ax.set_title(f"{port_name.capitalize()} Port - H3 Tiles")
    plt.tight_layout()
    
    # Save figure if output_path is specified
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved visualization to {output_path}")
    
    plt.show()

def visualize_satellite_image(image_path, output_path=None):
    """
    Visualize a satellite image.
    
    Args:
        image_path: Path to the image file
        output_path: Path to save the visualization
        
    Returns:
        None
    """
    try:
        img = Image.open(image_path)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Display image
        ax.imshow(img)
        
        # Add title
        h3_index = os.path.splitext(os.path.basename(image_path))[0]
        ax.set_title(f"Satellite Image - Tile {h3_index}")
        
        # Remove axes
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save figure if output_path is specified
        if output_path:
            plt.savefig(output_path, dpi=300)
            print(f"Saved visualization to {output_path}")
        
        plt.show()
    except Exception as e:
        print(f"Error visualizing image {image_path}: {e}")

def visualize_search_results(query_image_path, results, imagery_dir, output_path=None):
    """
    Visualize search results.
    
    Args:
        query_image_path: Path to the query image
        results: Search results from Qdrant
        imagery_dir: Directory containing imagery
        output_path: Path to save the visualization
        
    Returns:
        None
    """
    # Load query image
    query_img = Image.open(query_image_path)
    
    # Create figure
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results + 1, figsize=((n_results + 1) * 4, 4))
    
    # Show query image
    axes[0].imshow(query_img)
    query_h3_index = os.path.splitext(os.path.basename(query_image_path))[0]
    axes[0].set_title(f"Query: {query_h3_index}")
    axes[0].axis('off')
    
    # Show result images
    for i, result in enumerate(results):
        h3_index = result.payload['h3_index']
        port = result.payload['port']
        score = result.score
        
        # Load result image
        result_image_path = os.path.join(imagery_dir, port, f"{h3_index}.tif")
        if os.path.exists(result_image_path):
            result_img = Image.open(result_image_path)
            axes[i+1].imshow(result_img)
        else:
            axes[i+1].text(0.5, 0.5, "Image not found", ha='center', va='center')
        
        axes[i+1].set_title(f"Result {i+1}: {h3_index}\nPort: {port}\nScore: {score:.4f}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    
    # Save figure if output_path is specified
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved visualization to {output_path}")
    
    plt.show()

def visualize_similar_locations_on_map(query_coords, results, tiles_gdf, output_path=None):
    """
    Visualize similar locations on a map.
    
    Args:
        query_coords: Query coordinates [lat, lng]
        results: Search results from Qdrant
        tiles_gdf: GeoDataFrame with H3 tiles
        output_path: Path to save the visualization
        
    Returns:
        None
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot all tiles
    tiles_gdf.plot(ax=ax, color='none', edgecolor='gray', alpha=0.3)
    
    # Plot query point
    query_point = Point(query_coords[1], query_coords[0])  # [lng, lat]
    query_gdf = gpd.GeoDataFrame([{'geometry': query_point}], crs="EPSG:4326")
    query_gdf.plot(ax=ax, color='red', marker='*', markersize=100)
    
    # Plot result tiles
    for i, result in enumerate(results):
        h3_index = result.payload['h3_index']
        score = result.score
        
        # Get tile geometry
        tile_row = tiles_gdf[tiles_gdf['h3_index'] == h3_index]
        if len(tile_row) > 0:
            # Plot tile
            tile_row.plot(ax=ax, color='blue', alpha=0.5 * (1 - i/len(results)))
            
            # Add label
            centroid = tile_row.iloc[0]['geometry'].centroid
            ax.text(centroid.x, centroid.y, f"{i+1}: {score:.2f}", 
                    fontsize=10, ha='center', va='center', color='white',
                    bbox=dict(facecolor='blue', alpha=0.5))
    
    ax.set_title(f"Similar Locations to ({query_coords[0]:.4f}, {query_coords[1]:.4f})")
    plt.tight_layout()
    
    # Save figure if output_path is specified
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved visualization to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize satellite imagery and search results")
    parser.add_argument("--port", type=str, help="Port name for tile visualization")
    parser.add_argument("--tiles_path", type=str, default="../data/processed/all_tiles.geojson", help="Path to tiles GeoJSON file")
    parser.add_argument("--image_path", type=str, help="Path to satellite image for visualization")
    parser.add_argument("--output_dir", type=str, default="../data/visualizations", help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize port tiles
    if args.port:
        tiles_gdf = gpd.read_file(args.tiles_path)
        output_path = os.path.join(args.output_dir, f"{args.port}_tiles.png") if args.output_dir else None
        visualize_port_tiles(args.port, tiles_gdf, output_path)
    
    # Visualize satellite image
    if args.image_path:
        output_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.image_path))[0]}.png") if args.output_dir else None
        visualize_satellite_image(args.image_path, output_path)