# Circle Detection
# Programmer: Jenny Wang

import os
import numpy as np
import cv2


def nothing(x):
    # any operation here
    pass


cap = cv2.VideoCapture('lights.mp4')
font = cv2.FONT_HERSHEY_PLAIN

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # red
    lower_red1 = np.array([150, 50, 100])
    upper_red1 = np.array([255, 255, 255])
    lower_red2 = np.array([190, 50, 150])
    upper_red2 = np.array([255, 255, 255])
    # green
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])
    # yellow
    lower_yellow = np.array([15, 150, 150])
    upper_yellow = np.array([35, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.add(mask1, mask2)

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.erode(mask_red, kernel)

    # contours detection
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if area > 200:
            cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 10)

            if 10 < len(approx) < 20:
                print("red circle")
                cv2.putText(frame, "Red: STOP", (x + 1, y + 1), font, 2, (0, 255, 0), 2)

    for cnt in contours_green:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if area > 200:
            cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 10)

            if 10 < len(approx) < 20:
                print("green circle")
                cv2.putText(frame, "Green: GO", (x + 1, y + 1), font, 2, (0, 255, 0), 2)

    for cnt in contours_yellow:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if area > 200:
            cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 10)

            if 10 < len(approx) < 20:
                print("yellow circle")
                cv2.putText(frame, "Yellow: WAIT", (x + 1, y + 1), font, 2, (0, 255, 0), 2)

    cv2.imshow("Red Mask", mask_red)
    cv2.namedWindow("Red Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Red Mask", 250, 150)
    cv2.moveWindow("Red Mask", 0, 0)

    cv2.imshow("Green Mask", mask_green)
    cv2.namedWindow("Green Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Green Mask", 250, 150)
    cv2.moveWindow("Green Mask", 250, 0)

    cv2.imshow("Yellow Mask", mask_green)
    cv2.namedWindow("Yellow Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Yellow Mask", 250, 150)
    cv2.moveWindow("Yellow Mask", 0, 180)

    cv2.imshow("Main", frame)
    cv2.namedWindow("Main", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Main', 500, 300)
    cv2.moveWindow("Main", 0, 350)

    if cv2.waitKey(40) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


