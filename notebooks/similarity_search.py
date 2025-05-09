#!/usr/bin/env python
# coding: utf-8

# # Satellite Imagery Similarity Search
# 
# This notebook demonstrates how to perform similarity search on satellite imagery embeddings. You can search for similar locations based on:
# 
# 1. A specific location (coordinates)
# 2. An example tile
# 3. Multiple example locations (to find similar patterns)

# In[1]:


import os
import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image
import h3
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Add parent directory to path
sys.path.append('..')

from utils.ports import get_port_coordinates, get_port_bbox, get_all_ports
from utils.tiling import point_to_h3, h3_to_polygon
from utils.embeddings import generate_embedding_for_image
from utils.indexing import setup_vector_db, search_similar_tiles, create_filter_by_port, create_filter_by_location
from utils.visualization import visualize_search_results, visualize_similar_locations_on_map


# ## Configuration
# 
# Set the paths to your data directories and other parameters.

# In[ ]:


# Data directories
DATA_DIR = "../data"
IMAGERY_DIR = os.path.join(DATA_DIR, "raw")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "processed", "vector_db")
TILES_PATH = os.path.join(DATA_DIR, "processed", "all_tiles.geojson")

# Vector search parameters
COLLECTION_NAME = "satellite_embeddings"
TOP_K = 5  # Number of similar tiles to retrieve


# ## Load Data
# 
# Load the tiles and set up the vector database.

# In[ ]:


# Load tiles
tiles_gdf = gpd.read_file(TILES_PATH)
print(f"Loaded {len(tiles_gdf)} tiles")

# Set up vector database
client = QdrantClient(path=VECTOR_DB_PATH)

# Get collection info
collection_info = client.get_collection(COLLECTION_NAME)
print(f"Collection: {COLLECTION_NAME}")
print(f"Vector size: {collection_info.config.params.vectors.size}")
print(f"Distance: {collection_info.config.params.vectors.distance}")
print(f"Points count: {collection_info.points_count}")


# ## Search by Location
# 
# Search for similar locations based on coordinates.

# In[ ]:


def search_by_location(lat, lng, top_k=TOP_K, port_filter=None):
    """Search for similar locations based on coordinates."""
    # Convert to H3 cell
    h3_index = h3.latlng_to_cell(lat, lng, 4)
    print(f"H3 index: {h3_index}")
    
    # Find the corresponding tile in the database
    filter_condition = None
    if port_filter:
        filter_condition = create_filter_by_port(port_filter)
    
    # Search for the H3 index in the database
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="h3_index",
                    match=qmodels.MatchValue(value=h3_index)
                )
            ]
        ),
        limit=1
    )
    
    if not results.points:
        print(f"No tile found for H3 index {h3_index}")
        
        # Try to find the nearest tile
        nearest_filter = create_filter_by_location(lat, lng, 50)  # 50km radius
        nearest_results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=nearest_filter,
            limit=1
        )
        
        if not nearest_results.points:
            print("No nearby tiles found")
            return None
        
        print(f"Using nearest tile: {nearest_results.points[0].payload['h3_index']}")
        query_vector = nearest_results.points[0].vector
    else:
        query_vector = results.points[0].vector
    
    # Search for similar tiles
    search_results = search_similar_tiles(
        client=client,
        collection_name=COLLECTION_NAME,
        query_embedding=np.array(query_vector),
        top_k=top_k,
        filter_condition=filter_condition
    )
    
    return search_results

# Example: Search for locations similar to Tianjin Port
tianjin_coords = get_port_coordinates("tianjin")
results = search_by_location(tianjin_coords["lat"], tianjin_coords["lng"])

if results:
    # Print results
    print("\nSearch results:")
    for i, result in enumerate(results):
        h3_index = result.payload['h3_index']
        port = result.payload['port']
        score = result.score
        print(f"  {i+1}. {port.capitalize()} - Tile {h3_index}: Similarity score = {score:.4f}")
    
    # Visualize results on map
    visualize_similar_locations_on_map(
        query_coords=[tianjin_coords["lat"], tianjin_coords["lng"]],
        results=results,
        tiles_gdf=tiles_gdf
    )


# ## Search by Example Tile
# 
# Search for similar locations based on an example tile.

# In[ ]:


def search_by_example_tile(h3_index, port_name, top_k=TOP_K, port_filter=None):
    """Search for similar locations based on an example tile."""
    # Find the corresponding tile in the database
    filter_condition = None
    if port_filter:
        filter_condition = create_filter_by_port(port_filter)
    
    # Search for the H3 index in the database
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="h3_index",
                    match=qmodels.MatchValue(value=h3_index)
                ),
                qmodels.FieldCondition(
                    key="port",
                    match=qmodels.MatchValue(value=port_name)
                )
            ]
        ),
        limit=1
    )
    
    if not results.points:
        print(f"No tile found for H3 index {h3_index} in port {port_name}")
        return None
    
    query_vector = results.points[0].vector
    
    # Search for similar tiles
    search_results = search_similar_tiles(
        client=client,
        collection_name=COLLECTION_NAME,
        query_embedding=np.array(query_vector),
        top_k=top_k,
        filter_condition=filter_condition
    )
    
    return search_results

# Example: Get a random tile from Tianjin
tianjin_tiles = tiles_gdf[tiles_gdf['port'] == 'tianjin']
if len(tianjin_tiles) > 0:
    example_tile = tianjin_tiles.sample(1).iloc[0]
    example_h3_index = example_tile['h3_index']
    
    print(f"Example tile: {example_h3_index} (Tianjin)")
    
    # Search for similar tiles
    results = search_by_example_tile(example_h3_index, 'tianjin')
    
    if results:
        # Print results
        print("\nSearch results:")
        for i, result in enumerate(results):
            h3_index = result.payload['h3_index']
            port = result.payload['port']
            score = result.score
            print(f"  {i+1}. {port.capitalize()} - Tile {h3_index}: Similarity score = {score:.4f}")
        
        # Visualize results
        example_image_path = os.path.join(IMAGERY_DIR, 'tianjin', f"{example_h3_index}.tif")
        if os.path.exists(example_image_path):
            visualize_search_results(
                query_image_path=example_image_path,
                results=results,
                imagery_dir=IMAGERY_DIR
            )
        else:
            print(f"Image not found: {example_image_path}")


# ## Search by Multiple Examples
# 
# Search for locations similar to multiple example locations (e.g., grain elevators).

# In[ ]:


def search_by_multiple_examples(coordinates, top_k=TOP_K, port_filter=None):
    """Search for locations similar to multiple example locations."""
    # Convert coordinates to H3 cells
    h3_indices = [h3.latlng_to_cell(lat, lng, 4) for lat, lng in coordinates]
    print(f"H3 indices: {h3_indices}")
    
    # Find the corresponding tiles in the database
    query_vectors = []
    
    for h3_index in h3_indices:
        # Search for the H3 index in the database
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="h3_index",
                        match=qmodels.MatchValue(value=h3_index)
                    )
                ]
            ),
            limit=1
        )
        
        if results.points:
            query_vectors.append(np.array(results.points[0].vector))
    
    if not query_vectors:
        print("No tiles found for the provided coordinates")
        return None
    
    # Average the query vectors
    avg_query_vector = np.mean(query_vectors, axis=0)
    
    # Normalize the average vector
    norm = np.linalg.norm(avg_query_vector)
    if norm > 0:
        avg_query_vector = avg_query_vector / norm
    
    # Set up filter condition
    filter_condition = None
    if port_filter:
        filter_condition = create_filter_by_port(port_filter)
    
    # Search for similar tiles
    search_results = search_similar_tiles(
        client=client,
        collection_name=COLLECTION_NAME,
        query_embedding=avg_query_vector,
        top_k=top_k,
        filter_condition=filter_condition
    )
    
    return search_results

# Example: Search for locations similar to multiple example locations
# These could be coordinates of known grain elevators, for example
example_coordinates = [
    [39.02, 117.75],  # Example location 1
    [38.98, 117.82],  # Example location 2
    [39.01, 117.78]   # Example location 3
]

results = search_by_multiple_examples(example_coordinates)

if results:
    # Print results
    print("\nSearch results:")
    for i, result in enumerate(results):
        h3_index = result.payload['h3_index']
        port = result.payload['port']
        score = result.score
        print(f"  {i+1}. {port.capitalize()} - Tile {h3_index}: Similarity score = {score:.4f}")
    
    # Visualize results on map
    visualize_similar_locations_on_map(
        query_coords=example_coordinates[0],  # Use the first coordinate as the query point
        results=results,
        tiles_gdf=tiles_gdf
    )


# ## Custom Search
# 
# You can customize the search parameters and filters to find specific types of locations.

# In[ ]:


# Example: Search for locations similar to Tianjin Port, but only in Shanghai
tianjin_coords = get_port_coordinates("tianjin")
results = search_by_location(
    tianjin_coords["lat"], 
    tianjin_coords["lng"],
    port_filter="shanghai"
)

if results:
    # Print results
    print("\nSearch results (Shanghai only):")
    for i, result in enumerate(results):
        h3_index = result.payload['h3_index']
        port = result.payload['port']
        score = result.score
        print(f"  {i+1}. {port.capitalize()} - Tile {h3_index}: Similarity score = {score:.4f}")
    
    # Visualize results on map
    visualize_similar_locations_on_map(
        query_coords=[tianjin_coords["lat"], tianjin_coords["lng"]],
        results=results,
        tiles_gdf=tiles_gdf
    )

