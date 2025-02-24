import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

# Push-up counter variables
counter = 0
stage = None

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # Check if frame was read successfully
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks and calculate angle (only if a pose is detected)
        if results.pose_landmarks:  # Check if pose landmarks were detected
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates (no function)
                shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
                shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                elbow_x = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x
                elbow_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
                wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
               
                shoulder = np.array([shoulder_x, shoulder_y])
                elbow = np.array([elbow_x, elbow_y])
                wrist = np.array([wrist_x, wrist_y])

                # Calculate angle (no function)
                radians = np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0]) - np.arctan2(shoulder[1] - elbow[1], shoulder[0] - elbow[0])
                angle = np.abs(radians * 180.0 / np.pi)

                if angle > 180.0:
                    angle = 360 - angle

                # Visualize angle
                cv2.putText(image, str(int(angle)), tuple(np.multiply(elbow, [frame.shape[1], frame.shape[0]]).astype(int)),  # Use frame dimensions
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Push-up counter logic
                if angle > 160:  # When the angle is close to straight, user is in the "up" position
                    stage = "up"
                if angle < 90 and stage == 'up':  # When the angle is small, user is in the "down" position
                    stage = "down"
                    counter += 1
                    print(f"Push-up count: {counter}")

            except Exception as e:
                print(f"Error processing landmarks: {e}")  # Handle potential errors

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Display the final image
        cv2.imshow('Mediapipe Push-up Detection', image)

        # Exit on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()