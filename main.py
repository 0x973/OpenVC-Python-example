import cv2 as cv

print(cv.__version__)

done = False
# camera init
cap = cv.VideoCapture(0)

# Classifier init
eyeClassifier = cv.CascadeClassifier('haarcascade_eye.xml')
eyeClassifier.load('./haarcascades/haarcascade_eye.xml')
faceClassifier = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
faceClassifier.load('./haarcascades/haarcascade_frontalface_alt2.xml')

thickness = 2
numDetectionsTh = 10

while not done: 
    ret, frame = cap.read()

    # face detection
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    faceObjects = faceClassifier.detectMultiScale(gray)
    if faceObjects == () :
        cv.imshow('result', frame)
        key = cv.waitKey(1)
        done = key != -1 & key != 255
        continue

    for (x,y,h,w) in faceObjects:
        cv.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0), 
            (thickness)
        )

    # eyes detection
    eyeObjects = eyeClassifier.detectMultiScale(gray)
    for (x,y,h,w) in eyeObjects:
        cv.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0), 
            (thickness)
        )
        
    cv.imshow('result', frame)
    key = cv.waitKey(1)
    done = key != -1 & key != 255
