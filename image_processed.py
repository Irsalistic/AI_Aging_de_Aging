import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


import torch
import torchvision.transforms as T
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image

# Load the pre-trained DeepLabV3 model from torchvision
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()  # Set the model to evaluation mode


def process_image(image):
    # Convert numpy array to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Convert the image to RGB if it's a PIL Image
    if isinstance(image, Image.Image):
        image_rgb = np.array(image.convert("RGB"))  # Convert PIL Image to NumPy array
    else:
        image_rgb = image  # Assume it's already a NumPy array

    # Get the original size of the image
    original_size_cv = image_rgb.shape[:2]

    # Preprocess the image
    transform = T.Compose([
        T.Resize((520, 520)),  # Resize to the input size expected by the model
        T.ToTensor(),  # Convert image to a tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform segmentation
    with torch.no_grad():
        output = model(image_tensor)['out'][0]
    segmentation_map = output.argmax(0).cpu().numpy()

    # Post-process the result
    hair_class = 15  # Assuming the hair class is labeled as 15
    hair_segment = (segmentation_map == hair_class).astype(np.uint8)
    hair_segment_resized = Image.fromarray(hair_segment).resize(original_size_cv[::-1], Image.NEAREST)
    hair_segment_resized = np.array(hair_segment_resized)

    # Define landmark points
    eyes_nose = [27, 29, 30, 247, 226, 25, 110, 24, 23, 22, 26, 112, 244, 245, 188, 114, 217, 209, 49, 129, 64, 98, 97,
                 2, 326, 327, 358, 429, 437, 399, 412, 465, 464, 341, 256, 252, 253, 254, 339, 255, 446, 467, 260, 259,
                 257, 258, 286, 414, 413, 417, 168, 193, 189, 221, 28, 27]
    lips = [91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91]
    all_landmarks = eyes_nose + lips

    # Initialize MediaPipe face mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.1)

    # Detect face landmarks
    results = face_mesh.process(image_rgb)

    # Initialize a blank mask
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    # Check if landmarks are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_points = [tuple(np.array((face_landmarks.landmark[idx].x * original_size_cv[1],
                                               face_landmarks.landmark[idx].y * original_size_cv[0])).astype(int)) for
                               idx
                               in all_landmarks]
            # Draw the polygons for the landmark region
            cv2.fillPoly(mask, [np.array(landmark_points)], 255)

    # Resize landmark mask to match the original image size
    landmark_mask_resized = cv2.resize(mask, (original_size_cv[1], original_size_cv[0]), interpolation=cv2.INTER_NEAREST)

    # Apply the landmark mask to the hair segmented image
    hair_segment_resized[landmark_mask_resized == 255] = 0

    # Convert the numpy array to a format suitable for display with OpenCV
    hair_segment_resized = (hair_segment_resized * 255).astype(np.uint8)

    return hair_segment_resized