#!/usr/bin/env python3
"""
Test script for searching by multiple examples.

This script demonstrates how to search for similar locations based on multiple examples,
which is useful for finding specific types of infrastructure (e.g., grain elevators)
across different ports.
"""

import os
import sys
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import h3
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from utils.ports import get_port_coordinates, get_port_bbox, get_all_ports
from utils.tiling import h3_to_polygon
from utils.indexing import connect_to_qdrant

def search_by_multiple_examples(client, coordinates_list, limit=10, exclude_ports=None):
    """
    Search for similar locations based on multiple example coordinates.
    
    Args:
        client: Qdrant client
        coordinates_list: List of [lat, lon] coordinates
        limit: Maximum number of results to return
        exclude_ports: List of ports to exclude from search results
        
    Returns:
        List of search results
    """
    # Load a random embedding to use as a base
    embeddings_dir = "./data/embeddings"
    port_name = "tianjin"
    port_embeddings_dir = os.path.join(embeddings_dir, port_name)
    
    # Get all embedding files
    embedding_files = [f for f in os.listdir(port_embeddings_dir) if f.endswith('.npy')]
    
    if not embedding_files:
        print(f"No embeddings found for port: {port_name}")
        return None
    
    # Load a random embedding as a base
    base_h3_index = os.path.splitext(embedding_files[0])[0]
    base_embedding_path = os.path.join(port_embeddings_dir, f"{base_h3_index}.npy")
    base_embedding = np.load(base_embedding_path)
    
    # Convert coordinates to H3 indices
    h3_indices = []
    for lat, lon in coordinates_list:
        h3_index = h3.latlng_to_cell(lat, lon, 6)
        h3_indices.append(h3_index)
        print(f"Coordinates ({lat}, {lon}) are in H3 tile: {h3_index}")
    
    # Get embeddings for each H3 index
    embeddings = []
    
    for h3_index in h3_indices:
        # Try to load the embedding from file
        for port in get_all_ports():
            embedding_path = os.path.join(embeddings_dir, port, f"{h3_index}.npy")
            if os.path.exists(embedding_path):
                embedding = np.load(embedding_path)
                embeddings.append(embedding)
                print(f"Found embedding for H3 index {h3_index} in port {port}")
                break
        else:
            print(f"No embedding found for H3 index {h3_index}")
    
    if not embeddings:
        print("No embeddings found for the provided coordinates")
        return None
    
    # Average the embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    
    # Create filter if needed
    filter_obj = None
    if exclude_ports:
        must_not = []
        for port in exclude_ports:
            must_not.append(
                FieldCondition(
                    key="port",
                    match=MatchValue(value=port)
                )
            )
        filter_obj = Filter(must_not=must_not)
    
    # Search for similar locations
    search_results = client.search(
        collection_name="satellite_embeddings",
        query_vector=avg_embedding.tolist(),
        limit=limit,
        query_filter=filter_obj
    )
    
    return search_results

def visualize_results(results, query_coordinates=None):
    """
    Visualize search results on a map.
    
    Args:
        results: Search results from Qdrant
        query_coordinates: List of [lat, lon] coordinates used for the query
        
    Returns:
        Folium map
    """
    # Create a map centered on China
    m = folium.Map(location=[35, 120], zoom_start=5)
    
    # Add query points if provided
    if query_coordinates:
        for lat, lon in query_coordinates:
            folium.Marker(
                location=[lat, lon],
                popup="Query Point",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
    
    # Add results to the map
    for i, result in enumerate(results):
        # Get the H3 polygon
        h3_index = result.payload["h3_index"]
        h3_polygon = h3_to_polygon(h3_index)
        
        # Get the center of the H3 cell
        center = h3.cell_to_latlng(h3_index)
        lat, lon = center[0], center[1]
        
        # Add the polygon to the map
        folium.GeoJson(
            h3_polygon,
            style_function=lambda x, score=result.score: {
                "fillColor": "green",
                "color": "green",
                "weight": 2,
                "fillOpacity": min(0.8, max(0.2, score))
            },
            tooltip=f"Rank: {i+1}, Score: {result.score:.4f}, Port: {result.payload['port']}"
        ).add_to(m)
        
        # Add a marker for the center of the tile
        folium.Marker(
            location=[lat, lon],
            popup=f"Rank: {i+1}, Score: {result.score:.4f}, Port: {result.payload['port']}",
            icon=folium.Icon(color="blue", icon=str(i+1))
        ).add_to(m)
    
    # Save the map to an HTML file
    map_path = "data/processed/search_results_map.html"
    m.save(map_path)
    print(f"Map saved to {map_path}")
    
    return m

def main():
    """
    Main function to test multi-example search.
    """
    print("Testing multi-example search functionality...")
    
    # Connect to vector database
    vector_db_path = "./data/processed/vector_db"
    client = connect_to_qdrant(vector_db_path)
    print(f"Connected to vector database at {vector_db_path}")
    
    # Get collection info
    try:
        collection_info = client.get_collection("satellite_embeddings")
        print(f"Collection: satellite_embeddings")
        print(f"Vector dimension: {collection_info.config.params.vectors.size}")
        print(f"Distance metric: {collection_info.config.params.vectors.distance}")
        print(f"Number of points: {collection_info.vectors_count}")
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return
    
    # Define example coordinates (e.g., grain elevators)
    grain_elevator_coordinates = [
        [39.0150, 117.7200],  # Tianjin grain terminal
        [38.9600, 121.6400],  # Dalian grain terminal
        [31.3500, 121.5000]   # Shanghai grain terminal
    ]
    
    print("\nSearching for locations similar to grain elevators...")
    results = search_by_multiple_examples(client, grain_elevator_coordinates, limit=15)
    
    if results:
        print("\nSearch results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.payload['port']} - {result.payload['h3_index']} (Score: {result.score:.4f})")
        
        # Visualize results
        visualize_results(results, grain_elevator_coordinates)
    else:
        print("No results found")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()