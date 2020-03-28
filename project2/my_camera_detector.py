import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw square on the frame
    cv2.rectangle(frame, (5, 5), (220, 220), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', cv2.flip(frame, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()