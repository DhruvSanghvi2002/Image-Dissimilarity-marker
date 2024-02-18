# This is the task of image processing where we have to find the differences between the two images


# This is done using following steps
# First find the difference between the images
# Calculating the treshold of the image
# Dilating the image --> We have run the loop 2 times
# Calculating the contours  and finding the contours with area greater than hundred
# Mapping the different images in the stack



#Importing the necessary libraries
import cv2
import numpy as np
import imutils

# Loading the two images -->Images are stored in the images folder
image1=cv2.imread("images/input1.png")
image2=cv2.imread("images/input2.png")
# Image Preprocessing
# First we will convert this image to grayscale image
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#Resizing the image

#Calculating the pixel-wise difference between the images
diff = cv2.absdiff(gray1, gray2)
cv2.imshow("diff(img1, img2)", diff)
#Getting the threshold image, here if the pixel difference <30:0, 255
thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Threshold", thresh)
# Dilation
kernel = np.ones((5,5), np.uint8) 
dilate = cv2.dilate(thresh, kernel, iterations=2) 
cv2.imshow("Dilate", dilate)

# Calculate contours
contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

for contour in contours:
    if cv2.contourArea(contour) > 100:
        # Calculate bounding box around contour
        x, y, w, h = cv2.boundingRect(contour)
        # Draw rectangle - bounding box on both images
        cv2.rectangle(image1, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.rectangle(image2, (x, y), (x+w, y+h), (255,0,0), 2)

# Show images with rectangles on differences
x = np.zeros((image1.shape[0],10,3), np.uint8)
result = np.hstack((image1, x, image2))
cv2.imshow("Differences", result)
# Calculate similarity score
total_pixels = gray1.shape[0] * gray1.shape[1]
different_pixels = np.count_nonzero(thresh)  # Using the thresholded difference image
similarity_score = 100 * (1 - (different_pixels / total_pixels))

# Print similarity score
print("Similarity Score: {:.2f}%".format(similarity_score))

cv2.waitKey(0)
cv2.destroyAllWindows()


