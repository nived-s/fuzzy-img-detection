import numpy as np
import cv2
from random import randint


"""

Read a fuzzy image detect it edges using adaptive threshold apply necessary filters
After detecting draw detected shape on new canvas with different colour filled
Find area of contours

"""



# reading working image
img = cv2.imread("fuzzy.png", 1) 
cv2.imshow("original image", img)

# converting to gray and apply blur
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# applying blur in x=5 and y=5 
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Adaptive threshold
# max pixel value = 255
# inversing thresh since bg is white
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 205, 1)

cv2.imshow("thresh", thresh)

# contour edge detection after filtering the img
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("total contours detected:",len(contours))

filtered = []
for c in contours:
	# ignoring small pixels
	if cv2.contourArea(c) < 1000:
		continue

	# if it is large enough append to filtered
	filtered.append(c)

print("number contours after filter:", len(filtered))


# drawing detected contours on new canvas

# canvas
objects = np.zeros([img.shape[0], img.shape[1], 3], 'uint8')

# looping through each filtered cantours
for c in filtered:
	# random color generated to fill contour shape
	col = (randint(0,255), randint(0,255), randint(0,255))
	# drawing contour
	cv2.drawContours(objects, [c], -1, col, -1)
	# finding area
	area = cv2.contourArea(c)
	print("contours area: ", area)

# displaying final images
cv2.imshow("final", objects)

cv2.waitKey(0)
cv2.destroyAllWindows()