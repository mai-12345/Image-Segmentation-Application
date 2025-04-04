
import cv2
import math
import streamlit as st
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.title(" Image Segmentation Application")

upload_image=st.file_uploader('Please Upload an Image....',type=['jpg','png','jpeg','webp'])

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# Performs resizing and showing the image
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  st.image(img, 'segmented Image', use_container_width = True)

if upload_image is not None:
    file_bytes = np.asarray(bytearray(upload_image.read()), dtype = np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, 'Uploaded Image', use_container_width = True)
    
    BG_COLOR = (192, 192, 192) # gray
    MASK_COLOR = (255, 255, 255) # white


    # Create the options that will be used for ImageSegmenter
    base_options = python.BaseOptions(model_asset_path="deeplab_v3.tflite")
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)
    # Create the image segmenter
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        # Loop through demo image(s)
        # for image_file_name in img:
             # Retrieve the masks for the segmented image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask
        # Generate solid color images for showing the output segmentation mask.
        image_data = mp_image.numpy_view()
        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR

        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
        output_image = np.where(condition, fg_image, bg_image)

        
        resize_and_show(output_image)

    
    with python.vision.ImageSegmenter.create_from_options(options) as segmenter:
        
        # Create the MediaPipe Image
      image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

      # Retrieve the category masks for the image
      segmentation_result = segmenter.segment(image)
      category_mask = segmentation_result.category_mask

      # Convert the BGR image to RGB
      image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_BGR2RGB)

      # Apply effects
      blurred_image = cv2.GaussianBlur(image_data, (55,55), 0)
      condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
      output_image = np.where(condition, image_data, blurred_image)

      resize_and_show(output_image)
