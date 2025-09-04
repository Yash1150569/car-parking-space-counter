import cv2
import numpy as np
import cvzone
import pickle

width, height = (508 - 400), (235 - 189)

try:
    with open('carParkPos', 'rb') as f:
        poslist = pickle.load(f)
except:
    poslist = []

def mouseclick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        poslist.append((x, y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(poslist):
            x1, y1 = pos
            if x1 < x < (x1 + width) and y1 < y < (y1 + height):
                poslist.pop(i)
    with open('carParkPos', 'wb') as f:
        pickle.dump(poslist, f)

def checkParkingSpace(vidProc):
    spaceCounter = 0
    for pos in poslist:
        x, y = pos
        vidCrop = vidProc[y:y + height, x:x + width]
        count = cv2.countNonZero(vidCrop)
        if count < 500:
            spaceCounter += 1
            color = (0, 255, 0)
            thickness = 3
        else:
            color = (0, 0, 255)
            thickness = 1
        cvzone.putTextRect(vid, str(count), (x, y + height - 2), scale=1, thickness=1, offset=0, colorR=color)
        cv2.rectangle(vid, pos, (pos[0] + width, pos[1] + height), color, thickness)
    cvzone.putTextRect(vid, f'Free Space {spaceCounter}/{len(poslist)}', (450, 50), scale=2, thickness=3, offset=20,
                       colorR=(255, 200, 0))

cap = cv2.VideoCapture('carPark.mp4')

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video

    success, vid = cap.read()
    if not success:
        break

    img = cv2.imread('carPark.png')
    cv2.rectangle(img, (400, 189), (508, 235), (255, 0, 255), 2)

    for pos in poslist:
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2)

    vidGray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    vidBlur = cv2.GaussianBlur(vidGray, (3, 3), 1)



    # Apply HOG descriptor
    hog = cv2.HOGDescriptor()
    vidHOG = hog.compute(vidBlur)

    # Apply LBP descriptor
    radius = 1
    n_points = 8 * radius
    lbp = cv2.calcHist([vidBlur], [0], None, [256], [0, 256])

    # Use adaptive thresholding
    vidThreshold = cv2.adaptiveThreshold(vidBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    vidMedian = cv2.medianBlur(vidThreshold, 5)

    # Dilation to broaden the edges
    kernel = np.ones((3, 3), np.uint8)
    vidDilate = cv2.dilate(vidMedian, kernel, iterations=1)

    # Projective transformation and alignment
    rows, cols = vidDilate.shape[:2]
    pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows * 1.5], [cols * 1.5, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    vidTransform = cv2.warpPerspective(vidDilate, M, (cols, rows))

    checkParkingSpace(vidTransform)

    cv2.imshow('parking', img)
    cv2.setMouseCallback('parking', mouseclick)
    cv2.imshow('carParking', vid)
    cv2.waitKey(1)

