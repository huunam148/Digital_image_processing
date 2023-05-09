import cv2
import numpy as np

image = cv2.imread("sudoku.jpg", 0)


def preprocess_image(image):
    gray = cv2.GaussianBlur(image, (11, 11), 0)

    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C |
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    # Inverting the image
    gray = cv2.bitwise_not(gray)

    # Dilating the image to fill up the "cracks" in lines
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    gray = cv2.dilate(gray, kernel)
    image = gray

    return gray


def preprocess_image_2(image):
    gray = cv2.GaussianBlur(image, (5, 5), 0)

    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C |
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    # Inverting the image
    gray = cv2.bitwise_not(gray)

    # Dilating the image to fill up the "cracks" in lines
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    gray = cv2.dilate(gray, kernel)
    image = gray
    return gray


def find_largest_blob(image):
    height, width = image.shape[:2]
    max_area = 0
    seed_point = (None, None)

    # Loop through the image
    for x in range(width):
        for y in range(height):
            if image.item(y, x) == 255:
                area = cv2.floodFill(image, None, (x, y), 64)
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)

    cv2.floodFill(image, None, seed_point, 255)

    for x in range(width):
        for y in range(height):
            if image.item(y, x) == 64:  # Hide anything that isn't the main feature
                cv2.floodFill(image, None, (x, y), 0)

    # Return the binary image with only the largest blob
    return image

def find_corners_of_largest_blob(image):
    proc_image = preprocess_image(image)
    largest_blob = find_largest_blob(proc_image)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(
        largest_blob, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour (which should correspond to the sudoku)
    largest_contour = max(contours, key=cv2.contourArea)
    # Approximate the corners of the contour using the Ramer-Douglas-Peucker algorithm
    epsilon = 0.05 * cv2.arcLength(largest_contour, True)
    corners = cv2.approxPolyDP(largest_contour, epsilon, True)
    return corners

def reorder(pnt):
    pnt = pnt.reshape(4, 2)
    pnt_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = pnt.sum(1)
    pnt_new[0] = pnt[np.argmin(add)]
    pnt_new[3] = pnt[np.argmax(add)]
    diff = np.diff(pnt, axis=1)
    pnt_new[1] = pnt[np.argmin(diff)]
    pnt_new[2] = pnt[np.argmax(diff)]

    return pnt_new

def extract_corners(cor):
    w = 450
    h = 450
    cor = reorder(corners)
    pts1 = np.float32(cor)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_Warp = cv2.warpPerspective(image, matrix, (w, h))
    return img_Warp

def find_empty_location(arr, l):
    for row in range(9):
        for col in range(9):
            if (arr[row][col] == 0):
                l[0] = row
                l[1] = col
                return True
    return False


def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes


proc_image = preprocess_image(image)
largest_blob = find_largest_blob(proc_image)
corners = find_corners_of_largest_blob(largest_blob)
# print('corners \n' ,corners)
# print('corners reorder \n' ,reorder(corners))
extract_cor = extract_corners(corners)
img_binary = preprocess_image_2(extract_cor)


cv2.imwrite('output.jpg', img_binary)

box = splitBoxes(img_binary)

count = 0
file = open("output.txt", "w")
for i in range(81):
    n = 0
    contours, hierarchy = cv2.findContours(
        image=box[i], mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # largest_contour = max(contours, key=cv2.contourArea)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (cv2.contourArea(cnt) > 50) and w < 30 and w > 10 and h > 20:
            n = n + 1
    # print(n)
    if (n == 0):
        #print("_", end=" ")
        file.write(" _")
    if (n == 1):
        #print("X", end=" ")
        file.write(" X")
    count += 1
    if count % 9 == 0:
        #print("")
        file.write("\n")
file.close()



cv2.waitKey(0)
cv2.destroyAllWindows()
