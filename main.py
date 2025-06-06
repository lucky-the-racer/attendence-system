import cv2
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace

# Paths
known_path = "images"
unknown_path = "unknown_faces"
attendance_file = "attendance.csv"

os.makedirs(unknown_path, exist_ok=True)

# Prepare known names
known_faces = []
student_names = []
for file in os.listdir(known_path):
    if file.endswith((".jpg", ".png")):
        known_faces.append(os.path.join(known_path, file))
        student_names.append(os.path.splitext(file)[0].lower())

# Prepare attendance CSV
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write("Student Name,Time,Date,Program\n")

marked_names = []
unknown_counter = 1

# Initialize webcam
cap = cv2.VideoCapture(0)
print("📷 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    try:
        # Use OpenCV face detector to count faces first (fast)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        num_faces = len(faces)

        if num_faces == 0:
            cv2.putText(frame, "No face detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        elif num_faces > 1:
            cv2.putText(frame, "Only 1 person allowed", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            print("⚠️ More than 1 face detected, skipping recognition.")

        else:
            # Exactly 1 face, proceed with recognition on original frame for accuracy
            result = DeepFace.find(img_path=frame, db_path=known_path, enforce_detection=False, model_name="VGG-Face", detector_backend="opencv")

            if len(result) > 0 and not result[0].empty:
                identity_path = result[0].iloc[0]["identity"]
                name = os.path.splitext(os.path.basename(identity_path))[0].lower()

                if name not in marked_names:
                    now = datetime.now()
                    date = now.strftime('%Y-%m-%d')
                    time = now.strftime('%H:%M:%S')
                    program = "The Creator"

                    with open(attendance_file, 'a') as f:
                        f.write(f"{name},{time},{date},{program}\n")

                    marked_names.append(name)
                    print(f"✅ {name} recognized - Attendance Marked")

                    cv2.putText(frame, f"{name.upper()} - Attendance Marked", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            else:
                # Unknown person detected
                now = datetime.now()
                date = now.strftime('%Y-%m-%d')
                time = now.strftime('%H:%M:%S')
                unknown_name = f"unknown_{unknown_counter}"
                filename = os.path.join(unknown_path, f"{unknown_name}.jpg")

                cv2.imwrite(filename, frame)

                with open(attendance_file, 'a') as f:
                    f.write(f"{unknown_name},{time},{date},Unknown Entry\n")

                print(f"🚨 Unknown face detected - Saved as {unknown_name}.jpg")

                unknown_counter += 1
                cv2.putText(frame, "UNKNOWN - Marked", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    except Exception as e:
        print("⚠️ Detection error:", str(e))

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Attendance system stopped.")
