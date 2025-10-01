#!/usr/bin/env python3
"""
Download a sample image for testing object detection
"""

import urllib.request
import os

def download_sample_image():
    """Download a sample image for object detection testing"""

    # Sample image URL (a group photo for face detection)
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    filename = "sample_image.jpg"

    print(f"📥 Downloading sample image from: {url}")

    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Successfully downloaded: {filename}")
        print(f"📍 Location: {os.path.abspath(filename)}")
        return filename
    except Exception as e:
        print(f"❌ Error downloading image: {e}")

        # Try alternative URL
        alt_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Albert_Einstein_Head.jpg/800px-Albert_Einstein_Head.jpg"
        print(f"\n📥 Trying alternative URL: {alt_url}")

        try:
            urllib.request.urlretrieve(alt_url, filename)
            print(f"✅ Successfully downloaded: {filename}")
            return filename
        except Exception as e2:
            print(f"❌ Error with alternative URL: {e2}")
            return None

if __name__ == "__main__":
    download_sample_image()

