import imutils
import numpy as np
import pickle
import cv2
import face_recognition

def main():
    encoding_file = "encoding1.pickle"

    # Load face encodings and names
    with open(encoding_file, "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]

    # Open the video stream
    video_url = 'http://localhost:8000/video'
    a = 0
    while True:
        try:
            print(a)
            a += 1
            cap = cv2.VideoCapture(video_url)

            if not cap.isOpened():
                print("Error opening video stream.")
                return

            # Read frame from the video stream
            ret, frame = cap.read()

            if not ret:
                print("Error reading frame from video stream.")
                break

            # Convert the frame to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame for faster processing
            rgb = imutils.resize(rgb, width=400)

            # Find face locations and encodings in the frame
            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)

            names = []
            for encoding in encodings:
                # Compare face encodings with known encodings
                matches = face_recognition.compare_faces(known_encodings, encoding)
                name = "Unknown"

                if True in matches:
                    matched_indices = [i for i, match in enumerate(matches) if match]
                    counts = {}

                    # Count the number of matches for each known face
                    for index in matched_indices:
                        name = known_names[index]
                        counts[name] = counts.get(name, 0) + 1

                    # Determine the most recognized face
                    name = max(counts, key=counts.get)

                names.append(name)

            # Draw bounding boxes and labels on the frame
            for (top, right, bottom, left), name in zip(boxes, names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Frame", frame)

            # Increase FPS by reducing the delay
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print("Error reading frame from video stream:", str(e))
            continue

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
