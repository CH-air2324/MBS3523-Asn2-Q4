import cv2
import mediapipe as mp
import time

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

# Function to calculate the distance between two points
def calculate_distance(a, b):
    return ((b[0] - a[0])*2 + (b[1] - a[1]))*0.5

# Initialize webcam
cam = cv2.VideoCapture(0)

# Initialize Mediapipe Hands model
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.6)

# Time variables for FPS calculation
t_old = 0
t_new = 0

# Variables for counting thumb raises
thumb_raised = False
thumb_lowered = False
counter = 0

while True:
    success, img = cam.read()
    if not success:
        break

    # Recolor image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Make detection
    results = hands.process(imgRGB)

    # Recolor back to BGR
    imgRGB.flags.writeable = True
    img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)

    # Extract landmarks and draw hand landmarks
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color=(125, 0, 125), thickness=2, circle_radius=4),
                                  mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
                                  )

            # Get landmark for thumb
            thumb = (handLandmarks.landmark[4].x, handLandmarks.landmark[4].y)

            # Determine thumb state
            if thumb[1] < handLandmarks.landmark[3].y:  # Adjust this threshold as needed
                if not thumb_raised:
                    thumb_raised = True
                    thumb_lowered = False
                    counter += 1  # Increment counter when thumb is raised
            else:
                if thumb_raised:
                    thumb_lowered = True
                    thumb_raised = False

    # Render counter
    cv2.putText(img, f"Thumb Raises: {counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Calculate and display FPS
    t_new = time.time()
    fps = 1 / (t_new - t_old)
    t_old = t_new
    cv2.putText(img, 'FPS: ' + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Thumb Raises Counter', img)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cam.release()
cv2.destroyAllWindows()