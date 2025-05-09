"""
Port definitions for satellite imagery analysis.

This module defines the coordinates and bounding boxes for ports of interest.
"""

# Port coordinates (approximate center points)
PORT_COORDINATES = {
    "tianjin": {
        "lat": 39.0,
        "lng": 117.8,
        "description": "Tianjin Port, one of the largest ports in northern China"
    },
    "dalian": {
        "lat": 38.9,
        "lng": 121.6,
        "description": "Dalian Port, major port in northeastern China"
    },
    "shanghai": {
        "lat": 31.2,
        "lng": 121.5,
        "description": "Shanghai Port, world's busiest container port"
    },
    "qingdao": {
        "lat": 36.1,
        "lng": 120.3,
        "description": "Qingdao Port, major port in eastern China"
    }
}

# Port bounding boxes [min_lng, min_lat, max_lng, max_lat]
# These are approximate and can be adjusted as needed
PORT_BBOXES = {
    "tianjin": [117.7, 38.95, 117.85, 39.05],
    "dalian": [121.5, 38.85, 121.7, 39.0],
    "shanghai": [121.3, 31.1, 121.6, 31.4],
    "qingdao": [120.2, 36.0, 120.4, 36.2]
}

def get_port_coordinates(port_name):
    """
    Get coordinates for a specific port.
    
    Args:
        port_name: Name of the port (tianjin, dalian, shanghai, qingdao)
        
    Returns:
        Dictionary with lat, lng, and description
    """
    port_name = port_name.lower()
    if port_name not in PORT_COORDINATES:
        raise ValueError(f"Unknown port: {port_name}. Available ports: {list(PORT_COORDINATES.keys())}")
    
    return PORT_COORDINATES[port_name]

def get_port_bbox(port_name):
    """
    Get bounding box for a specific port.
    
    Args:
        port_name: Name of the port (tianjin, dalian, shanghai, qingdao)
        
    Returns:
        List [min_lng, min_lat, max_lng, max_lat]
    """
    port_name = port_name.lower()
    if port_name not in PORT_BBOXES:
        raise ValueError(f"Unknown port: {port_name}. Available ports: {list(PORT_BBOXES.keys())}")
    
    return PORT_BBOXES[port_name]

def get_all_ports():
    """
    Get list of all available ports.
    
    Returns:
        List of port names
    """
    return list(PORT_COORDINATES.keys())