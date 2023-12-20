import numpy as np
import cv2
import matplotlib.pyplot as plt

# Source: https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/

img1 = cv2.imread("img1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("img2.jpg", cv2.IMREAD_GRAYSCALE)

# sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

# orb = cv2.ORB_create(nfeatures=1500)

keypoints1, descriptors1 = surf.detectAndCompute(img1, None)
img1 = cv2.drawKeypoints(img1, keypoints1, None)

keypoints2, descriptors2 = surf.detectAndCompute(img2, None)
img2 = cv2.drawKeypoints(img2, keypoints2, None)

bf = cv2.BFMatcher()
matches = bf.match(descriptors1,descriptors2)
matches = sorted(matches, key = lambda x:x.distance)


img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,matches[:10],None, flags=2)

plt.imshow(img3)
plt.show()

# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()