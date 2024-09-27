#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


min_contour_width = 40  # Minimum width of the contour to consider it a vehicle
min_contour_height = 40  # Minimum height of the contour to consider it a vehicle
offset = 10  # Offset for vehicle detection near the counting line
line_height = 550  # Position of the line where vehicles are counted
matches = []  # Stores centroid coordinates of detected vehicles
vehicles = 0  # Counter for total number of vehicles


# In[3]:


def get_centroid(x, y, w, h):
   x1 = int(w / 2)
   y1 = int(h / 2)
   cx = x + x1
   cy = y + y1
   return cx, cy


# In[4]:


cap = cv2.VideoCapture('Video.mp4')


# In[5]:


cap.set(3, 1920)  # Set width
cap.set(4, 1080)  # Set height

if cap.isOpened():
   ret, frame1 = cap.read()  # Read first frame
else:
   ret = False

ret, frame1 = cap.read()  # Read first frame
ret, frame2 = cap.read()  # Read second frame for frame differencing


# In[ ]:


import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('Video.mp4')  # Replace with the actual video path

# Read the first two frames from the video
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Ensure the frames are successfully read
if not ret:
    print("Error: Could not read video frames.")
    cap.release()
    exit()

# Main processing loop
while ret:
    # Frame differencing
    d = cv2.absdiff(frame1, frame2)  # Find the absolute difference between two frames
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    # Apply thresholding to get a binary image
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))  # Dilate the image to fill in gaps

    # Morphological closing to close gaps in contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Find contours in the binary image
    contours, h = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Update frames for the next iteration
    frame1 = frame2
    ret, frame2 = cap.read()  # Read the next frame

    # Break the loop if no more frames are available
    if not ret:
        break

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('Video.mp4')  # Replace with the actual video path

# Define constants for contour filtering
min_contour_width = 40  # Minimum contour width to consider as a vehicle
min_contour_height = 40  # Minimum contour height to consider as a vehicle
line_height = 550  # Y-coordinate of the line for vehicle counting
offset = 10  # Offset for counting margin
vehicles = 0  # Vehicle counter
matches = []  # Store centroids of vehicles

# Define the function to calculate the centroid of a bounding box
def get_centroid(x, y, w, h):
    cx = x + w // 2  # Calculate x-center
    cy = y + h // 2  # Calculate y-center
    return (cx, cy)

# Read the first two frames for frame differencing
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Ensure frames were read successfully
if not ret:
    print("Error: Could not read video frames.")
    cap.release()
    exit()

# Main processing loop
while ret:
    # Frame differencing
    d = cv2.absdiff(frame1, frame2)  # Find the absolute difference between two frames
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    # Apply thresholding to get binary image
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))  # Dilate the image to fill in gaps

    # Morphological closing to close gaps in contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Find contours in the binary image
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the counting line
    cv2.line(frame1, (0, line_height), (frame1.shape[1], line_height), (0, 255, 0), 2)

    # Process each contour
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)  # Get bounding box of the contour

        # Check if the contour is large enough to be a vehicle
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)
        if not contour_valid:
            continue

        # Draw a rectangle around detected vehicle
        cv2.rectangle(frame1, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)

        # Get centroid of the vehicle
        centroid = get_centroid(x, y, w, h)
        matches.append(centroid)
        cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

        # Count vehicles crossing the line
        cx, cy = centroid
        for (mx, my) in matches:
            if line_height - offset < my < line_height + offset:
                vehicles += 1
                matches.remove((mx, my))  # Remove vehicle from matches after counting

    # Display the vehicle count on the frame
    cv2.putText(frame1, f"Total Vehicles Detected: {vehicles}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)

    # Show the frame in real-time
    cv2.imshow('Vehicle Detection', frame1)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update frames for the next iteration
    frame1 = frame2
    ret, frame2 = cap.read()  # Read the next frame

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

