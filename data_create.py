import cv2
import os

# Define the gestures and corresponding directories to save images
# gestures = ["Volume", "Brightness", "Pointer", "Forward", "Backward", "Next", "Previous"]
gestures = ["No_gesture"]
base_dir = "recorded_images"

# Create directories for each gesture if they don't exist
for gesture in gestures:
    os.makedirs(os.path.join(base_dir, gesture), exist_ok=True)

# Function to record and save images for a given gesture
def record_gesture_images(gesture):
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Counter to keep track of saved images
    img_count = 0

    while True:
        # Capture frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Display the frame
        frame_copy = frame.copy()
        cv2.putText(frame_copy, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame_copy, "Press 's' to save the image", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow("Recording Gesture", frame_copy)

        # Press 's' to save the current frame as an image
        key = cv2.waitKey(1)
        if key == ord('s'):
            # Save the image with a unique filename
            img_name = f"{gesture}_{img_count}.jpg"
            img_path = os.path.join(base_dir, gesture, img_name)
            cv2.imwrite(img_path, frame)
            print(f"Saved image: {img_path}")
            img_count += 1

        # Press 'q' to exit
        elif key == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Record images for each gesture
for gesture in gestures:
    print(f"Recording images for gesture: {gesture}")
    record_gesture_images(gesture)

print("Recording complete.")
