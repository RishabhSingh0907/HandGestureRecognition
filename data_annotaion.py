import os
import cv2
from hand_utils import hand_detect, NoHandsDetectedError

# Path to your curated dataset directory
dataset_dir = r"C:\Users\risha\Program_folder\Computer_Vision\Gesture_recognition\recorded_images"

# Directory to save detected hand images
output_dir = "detected_hands"
os.makedirs(output_dir, exist_ok=True)

# CSV file to store hand coordinates and gestures
csv_file = open("hand_coordinates.csv", "w")
csv_file.write("Image,Gesture\n")

# Process each subfolder (gesture) in the curated dataset directory
for gesture_folder in os.listdir(dataset_dir):
    gesture_path = os.path.join(dataset_dir, gesture_folder)
    if os.path.isdir(gesture_path):
        gesture_name = gesture_folder.lower()  # Get the gesture name from the folder name

        # Skip hand detection for folders named "No_gesture"
        if gesture_name.lower() == "no_gesture":
            # Process each image in the "No_gesture" folder
            for file_name in os.listdir(gesture_path):
                if file_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.relpath(os.path.join(gesture_path, file_name), os.getcwd())
                    # Write image path with zeros for coordinates and "NoGesture" as the gesture
                    csv_file.write(f"{image_path},NoGesture\n")
                    print(f"Labeling as NoGesture: {image_path}")

        else:
            # Process each image in the gesture folder
            for file_name in os.listdir(gesture_path):
                if file_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.relpath(os.path.join(gesture_path, file_name), os.getcwd())
                    print(image_path)
                    print(type(image_path))
                    try:
                        img = cv2.imread(image_path)
                        # Detect hands in the image and get hand coordinates
                        xmin, xmax, ymin, ymax = hand_detect(img=img)

                        # Extract image name
                        image_name = os.path.splitext(file_name)[0]
                        
                        # Save the detected hand image
                        output_path = os.path.join(output_dir, f"{image_name}.jpg")
                        detected_hand_image = img[ymin:ymax, xmin:xmax]
                        cv2.imwrite(output_path, detected_hand_image)

                        # Write hand coordinates and gesture to CSV file
                        csv_file.write(f"{output_path},{gesture_name}\n")

                        print(f"Hand detected in: {image_path}")

                    except NoHandsDetectedError:
                        # If no hands are detected
                        print(f"No hands detected in: {image_path}")
                        continue

                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
                        continue

# Close the CSV file
csv_file.close()

print("Hand detection and CSV creation complete.")
