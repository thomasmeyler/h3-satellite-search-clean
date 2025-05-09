#!/usr/bin/env python3
"""
Save Google Earth Engine credentials to a file.
"""

import json
import os

def save_credentials():
    """
    Save Google Earth Engine credentials to a file.
    """
    credentials = {
        "type": "service_account",
        "project_id": "your-project-id",
        "private_key_id": "your-private-key-id",
        "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n",
        "client_email": "your-service-account@your-project.iam.gserviceaccount.com",
        "client_id": "your-client-id",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com",
        "universe_domain": "googleapis.com"
    }
    
    # Save credentials to file
    with open('gee_credentials.json', 'w') as f:
        json.dump(credentials, f, indent=2)
    
    print(f"Credentials saved to {os.path.abspath('gee_credentials.json')}")
    
    # Test authentication
    try:
        import ee
        with open('gee_credentials.json', 'r') as f:
            creds = json.load(f)
        credentials = ee.ServiceAccountCredentials(
            creds['client_email'], 
            'gee_credentials.json'
        )
        ee.Initialize(credentials)
        print("Authentication successful!")
        
        # Test a simple Earth Engine operation
        image = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20230101T025941_20230101T030534_T51RUQ')
        print(f"Image info: {image.getInfo()['properties']['system:index']}")
        
    except Exception as e:
        print(f"Authentication failed: {e}")

if __name__ == "__main__":
    save_credentials()