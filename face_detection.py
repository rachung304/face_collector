import cv2
import argparse
import datetime
import time
import json
import io
import dropbox
import warnings
import numpy
import imutils
import os
from picamera.array import PiRGBArray
from picamera import PiCamera
from DBUploader.tempimage import TempImage 

#Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="/home/pi/conf.json")
args = vars(ap.parse_args())

warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None


# Create memory stream
stream = io.BytesIO()

#Get the picture (low res to increase speed)
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.capture(stream, format='jpeg')
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

# check dropbox access
if conf["use_dropbox"]:
    client = dropbox.Dropbox(conf["dropbox_access_token"])
    print("[SUCCESS] dropbox account linked!")
else:
    print("[Failed] Dropbox account not found")
        
#Load a cascade file for detecting faces
face_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')

print("[INFO] Warming Camera Up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format = "bgr", use_video_port=True):
    # Grab the raw NumPy array representing the image and initialize
    # the timestemp and occupied/unoccupied text
    frame = f.array
    timestamp = datetime.datetime.now()
    text = "Unoccupied"

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0 )

    if avg is None:
        print("[INFO] Starting background model...")
        avg = gray.copy().astype("float")
        rawCapture.truncate(0)
        continue

    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh,None, iterations = 2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loops over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        text = "Occupied"

    # draw the text and timestamp on the frame
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, "Room Status: {}".format(text), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255),1)

    # check to see if room is occupied
    if text == "Occupied":
        # check to see if enough time has passed between uploads
        if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
            motionCounter += 1

            # check to see if the number of frames with consistent motion is high enough
            if motionCounter >= conf["min_motion_frames"]:


                #Look for faces in the image using the loaded cascade file
                faces = face_cascade.detectMultiScale(frame, 1.3, 5)
                print("Found " + str(len(faces)) + " face(s)")

                #Draw a rectangle around every found face
                if len(faces) >= 1:
                    for (x,y,w,h) in faces:
                        crop_img = frame[y:y+h, x:x+w]
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                        cv2.imshow('cropped', crop_img)
                        t = TempImage()
                        cv2.imwrite(t.path, crop_img)
                        print("[UPLOAD] {}".format(ts))
                        path = "/{base_path}/{timestamp}.jpg".format(base_path=conf["dropbox_base_path"], timestamp=ts)
                        client.files_upload(open(t.path, "rb").read(), path)
                        t.cleanup()

                lastUploaded = timestamp
                motionCounter = 0
        else:
            motionCounter = 0

    if conf["show_video"]:
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break
    rawCapture.truncate(0)
            
        
'''
#Convert the picture into numpy array
buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

#Create OpenCV image
image = cv2.imdecode(buff, 1)

#Load a cascade file for detecting faces
face_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')

#Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Look for faces in the image using the loaded cascade file
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print("Found " + str(len(faces)) + " face(s)")

#Draw a rectangle around every found face
for (x,y,w,h) in faces:
    crop_img = image[y:y+h, x:x+w]
    cv2.imshow('cropped', crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,255,0),2)
    
#Save resulting image
img = cv2.imwrite('result.jpg', image)
resimg = cv2.resize(img, (960,540))
cv2.imshow('FRAME', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
