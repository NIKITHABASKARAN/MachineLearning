import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time  # Import for timer functionality

# Initialize deques for different colors
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

# Initialize the color index and kernel
colorIndex = 0  # 0: Blue, 1: Green, 2: Red, 3: Yellow
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
kernel = np.ones((5, 5), np.uint8)

# Create a blank white canvas for drawing
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)  # Clear Button
cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), -1)  # Blue
cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), -1)  # Green
cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), -1)  # Red
cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), -1)  # Yellow

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# Initialize MediaPipe hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize variables for CLEAR message display
show_clear_message = False
clear_message_start_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Process the hand landmarks if detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark]
            index_finger_tip = landmarks[8]  # Get the tip of the index finger (ID 8)

# Draw a circle at the index finger tip in the chosen color
            cv2.circle(frame, index_finger_tip, 10, colors[colorIndex], -1)
            # Check for button presses
            if index_finger_tip[1] <= 65:  # Finger near the top for button press
                if 40 <= index_finger_tip[0] <= 140:  # Clear button
                    show_clear_message = True
                    clear_message_start_time = time.time()  # Record the start time
                elif 160 <= index_finger_tip[0] <= 255:  # Blue button
                    colorIndex = 0
                elif 275 <= index_finger_tip[0] <= 370:  # Green button
                    colorIndex = 1
                elif 390 <= index_finger_tip[0] <= 485:  # Red button
                    colorIndex = 2
                elif 505 <= index_finger_tip[0] <= 600:  # Yellow button
                    colorIndex = 3
            else:
                # Draw on the canvas based on finger movement
                if colorIndex == 0:
                    bpoints[-1].appendleft(index_finger_tip)
                elif colorIndex == 1:
                    gpoints[-1].appendleft(index_finger_tip)
                elif colorIndex == 2:
                    rpoints[-1].appendleft(index_finger_tip)
                elif colorIndex == 3:
                    ypoints[-1].appendleft(index_finger_tip)

    # Display "CLEARED!" message and reset canvas after 2 seconds
    if show_clear_message:
        elapsed_time = time.time() - clear_message_start_time
        if elapsed_time < 2:  # Show "CLEARED!" for 2 seconds
            cv2.putText(paintWindow, "CLEARED!", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
        else:
            show_clear_message = False
            paintWindow[67:, :, :] = 255  # Clear the drawing area
            bpoints = [deque(maxlen=512)]  # Reset all drawing points
            gpoints = [deque(maxlen=512)]
            rpoints = [deque(maxlen=512)]
            ypoints = [deque(maxlen=512)]

    # Draw lines on the canvas for each color
    for i, points in enumerate([bpoints, gpoints, rpoints, ypoints]):
        for j in range(len(points)):
            for k in range(1, len(points[j])):
                if points[j][k - 1] is None or points[j][k] is None:
                    continue
                cv2.line(frame, points[j][k - 1], points[j][k], colors[i], 2)
                cv2.line(paintWindow, points[j][k - 1], points[j][k], colors[i], 2)

    # Show the webcam frame and the canvas
    cv2.imshow("Hand Drawing", frame)
    cv2.imshow("Canvas", paintWindow)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
