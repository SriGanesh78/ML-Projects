# Fix for Google Colab Import Error

## Problem

The notebook was trying to use `google.colab.patches.cv2_imshow` which only works in Google Colab, not on local Jupyter.

## Solution Applied

I've already fixed the imports in your notebook. The `cv2_imshow` function has been replaced with a matplotlib-based alternative.

## Image File Issue

The code was trying to read `img.jpeg` which doesn't exist. I've downloaded a sample image for you.

### Option 1: Use the Sample Image (Recommended)

Replace this line in your notebook:

```python
img = cv2.imread("img.jpeg")
```

With:

```python
img = cv2.imread("sample_image.jpg")
```

### Option 2: Use Your Own Image

1. Put any `.jpg` or `.png` image in the same folder as your notebook
2. Update the code to use your image filename:

```python
img = cv2.imread("your_image_name.jpg")
```

### Option 3: Use Webcam (Real-time Detection)

Replace the image loading code with webcam capture:

```python
# Use webcam for real-time detection
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 1)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img[y:y+h, x:x+w], (ex,ey), (ex+ew,ey+eh), (255,0,0), 3)

    # Display the image
    cv2.imshow('Face Detection', img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Quick Fix Code for Your Notebook

Add this cell at the beginning of your face detection code:

```python
import cv2
import matplotlib.pyplot as plt
import os

# Check if image exists, if not download sample
if not os.path.exists("img.jpeg") and not os.path.exists("sample_image.jpg"):
    import urllib.request
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    urllib.request.urlretrieve(url, "sample_image.jpg")
    print("✅ Downloaded sample image")

# Use sample image if img.jpeg doesn't exist
image_file = "img.jpeg" if os.path.exists("img.jpeg") else "sample_image.jpg"
img = cv2.imread(image_file)

if img is None:
    print(f"❌ Error: Could not read image file: {image_file}")
else:
    print(f"✅ Successfully loaded: {image_file}")
    print(f"📐 Image size: {img.shape}")
```

## What's Been Fixed

✅ Google Colab imports removed
✅ Custom `cv2_imshow()` function added
✅ Sample image downloaded

## What You Need to Do

📝 Update the image filename in your code from `"img.jpeg"` to `"sample_image.jpg"`

or

📝 Add your own image to the project folder

Your notebook should now work perfectly! 🎉
