import numpy as np
import cv2
from matplotlib import pyplot as plt
import xlrd


def get_coordinates():
    loc = ("C:/Users/Kartikaeya/Desktop/projects/coordinates.xlsx")
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)
    coordinates = np.zeros((sheet.nrows, 2))
    for i in range(sheet.nrows):
        coordinates[i, 0] = sheet.cell_value(i, 0)
        coordinates[i, 1] = sheet.cell_value(i, 1)
    return coordinates.astype(int)

def draw_rectangles(frame, coordinates):
    length = coordinates.shape[0]
    for i in range(length):
        cv2.rectangle(frame, (coordinates[i, 0], coordinates[i, 1]), (coordinates[i, 0] + 10, coordinates[i, 1] + 10),
                      [0,0,255], 1)

def hand_histogram(frame, coordinates):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([340, 10, 3], dtype=hsv_frame.dtype)
    for i in range(34):
        roi[i * 10: i * 10 + 10, 0:10, :] = hsv_frame[coordinates[i, 1]:(coordinates[i, 1] + 10),
                                            coordinates[i, 0]:coordinates[i, 0] + 10, :]
    hand_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    return (cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX))

def hist_masking(frame, hand_hist):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dst = cv2.calcBackProject([hsv_frame], [0, 1], hand_hist, [0, 180, 0, 256], 1)
    # disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

    # cv2.filter2D(dst, -1, disc, dst)
    cv2.imshow('dst before blurring ', dst)
    thresh = cv2.medianBlur(dst, 5)
    thresh = cv2.erode(thresh, None, iterations=5)
    thresh = cv2.dilate(thresh, None, iterations=5)
    cv2.imshow('dst after threshing', thresh )
    thresh = cv2.merge((thresh, thresh, thresh))
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('after threshing', thresh)
    return (thresh)

def main():
    is_hand_histogram_created = False
    coordinates = get_coordinates()
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        pressed_key = cv2.waitKey(1)
        ret , frame = cap.read()
        frame = cv2.flip(frame , 1)
        if pressed_key ==27:
            break
        if pressed_key & 0xFF == ord("s"):
            hand_hist = hand_histogram(frame, coordinates)
            is_hand_histogram_created = True
        if is_hand_histogram_created:
            thresh = hist_masking(frame, hand_hist)
        else:
            draw_rectangles(frame, coordinates)
        cv2.imshow('frame', frame)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
