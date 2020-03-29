import numpy as np 
import cv2

faceCascade = cv2.CascadeClassifier('cascade_watch_10.xml')

video_capture = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    watches = faceCascade.detectMultiScale(
        gray, 50, 50)

    # Draw a rectangle around the faces
    for (x, y, w, h) in watches:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
