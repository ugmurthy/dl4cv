# to demonstrate how a image read by open CV imread can be converted to
# PIL image and then resized.

import cv2
import imutils
import numpy as np
from PIL import Image

camera = cv2.VideoCapture(0);
frame_no = 0
while (True):
    (grabbed, frame) = camera.read()
    frame = imutils.resize(frame, width=600)
    frame_no += 1


    # convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # convert to a PIL IMAGE
    frame_pil = Image.fromarray(frame)
    # resize to a square size
    frame_rpil = frame_pil.resize((299,299))
    # convert back to numpy array for display
    frame = np.asarray(frame_rpil)
    # Convert back to BGR for display
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.putText(frame, "Frame: {}".format(frame_no), (10,30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
    cv2.imshow("Video",frame)
    #################################################
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break

camera.release()
cv2.destroyAllWindows()
