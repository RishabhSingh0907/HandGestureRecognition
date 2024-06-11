import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode = False,
    model_complexity = 0, 
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.6,
    max_num_hands = 2
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Class for handling exceptions when no hands are detected~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class NoHandsDetectedError(Exception):
    def __init__(self, message="No hands were detected."):
        self.message = message
        super().__init__(self.message)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Funciton for extract landmarks of the hands from the image~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
def hand_landmark(img):
    # img = cv2.imread(image_path)
    RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(RGBimg)
    return results

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Funciton for detection the hands from the image~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def hand_detect(img): 
    padding = 30
    result = hand_landmark(img)
    h, w, c = img.shape
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmark_x = [lm.x for lm in hand_landmarks.landmark]
            landmark_y = [lm.y for lm in hand_landmarks.landmark]

            x_min = min(landmark_x)
            x_max = max(landmark_x)
            y_min = min(landmark_y)
            y_max = max(landmark_y)

            # Convert bounding box coordinate to integer
            x_min = int(x_min * w)
            x_max = int(x_max * w)
            y_min = int(y_min * h)
            y_max = int(y_max * h)

            # Add padding to the coordinates
            x_min -= padding 
            x_max += padding
            y_min -= padding 
            y_max += padding

            # Ensure bounding box coordinates are withing the range of the image
            x_min_resized = max(x_min, 0)
            x_max_resized = min(x_max, w)
            y_min_resized = max(y_min, 0)
            y_max_resized = min(y_max, h)
            
            # returning detected hand image with padding to show full hand
            return x_min_resized, x_max_resized, y_min_resized, y_max_resized
    else:
        raise NoHandsDetectedError("No hands were detected in the frame")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Locate and draw the hand landmark over the image ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def draw_landmark_connections(img):
    result = hand_landmark(img)
    if result.multi_hand_landmarks:
        connections = mp_hands.HAND_CONNECTIONS
        mp_drawing.draw_landmarks(
            img, result.multi_hand_landmarks[0], connections,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        return img
    else:
        raise NoHandsDetectedError("No hands were detected in the frame")
