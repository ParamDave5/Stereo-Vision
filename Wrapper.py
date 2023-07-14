import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import time

from utils.misc import *

parser = argparse.ArgumentParser(description = 'Path to input images')
parser.add_argument('--Path' , default= '/Users/sheriarty/Desktop/enpm673/enpm673Proj3/data/' , help = 'path to data folder')
parser.add_argument('--Pair' , default = 'pendulum' , help = 'subfolder in data folder')
parser.add_argument('--display' , default = "False" , help = 'to display images or not')

args = parser.parse_args()
paths = args.Path
pair = args.Pair
display = args.display
path = paths + pair 

images = [cv2.imread(file) for file in glob.glob(path + "/*.png")]
gray = grayImages(images )
img1 , img2 = images[0] , images[1]
h1,w1 = img1.shape[:2]
h2,w2 = img2.shape[:2]

keypoints1 , descriptors1 = sift(images[0] , display)
keypoints2 , descriptors2 = sift(images[1] ,display)

points1 , points2 = featureMatching(keypoints1 , descriptors1 , keypoints2 ,descriptors2)
points1 , points2 = np.array(points1) , np.array(points2)

F , pts1_inliers , pts2_inliers = ransac(points1 , points2 , 0.001 , 2500)
print("Value of F using custom algorithm method is:",F)

F_cv2 , mask = cv2F(points1 , points2)
print("Value of F using inbuilt method is: " ,F_cv2)

k1 ,k2 , parameters = returnParams(pair)

E = getEssentialMatrix(k1 , k2 , F )
print('E for' + pair + " is :" , E)

# r , c = decomposeE(E)
# print(c)
R_final , C_final , pts3d = recoverPose(E,points1 , points2 , k1 , k2) 

l1 = cv2.computeCorrespondEpilines(points2.reshape(-1,1,2) , 2 , F)
epilines2 , _ = drawEpilines(images[0] , images[1] ,l1, points1 , points2)
l2 = cv2.computeCorrespondEpilines(points1.reshape(-1,1,2) , 2 , F)
epilines1,_ = drawEpilines(images[1] , images[0] , l2[:10] , points2 , points1)

stack = np.hstack((epilines1 , epilines2))
cv2.imwrite(pair+"epilines" +  ".png", stack)

plt.imshow(stack)
plt.show()

ret ,H1 , H2 = cv2.stereoRectifyUncalibrated(np.float32(points1) , np.float32(points2) , F , imgSize=(w1,h1))
print('H1 for' + pair + " is :" , H1)
print('H2 for' + pair + " is :" , H2)
image1Rect = cv2.warpPerspective(img1, H1 , (w1,h1))
image2Rect = cv2.warpPerspective(img2, H1 , (w2,h2))
stack1 = np.hstack((image1Rect , image2Rect))
cv2.imwrite(pair+"rectified" +  ".png", stack1)

plt.imshow(stack1)
plt.show()

dst1 = cv2.perspectiveTransform(points1.reshape(-1,1,2), H1).squeeze()
dst2 = cv2.perspectiveTransform(points2.reshape(-1,1,2), H2).squeeze()
rectLines1 = epilines(points2 ,2 , F,w2)
warpedlines1 = warpEpilines(rectLines1 , H1)
rectLines2 = epilines(points1 ,2 , F,w2)
warpedlines2 = warpEpilines(rectLines2 , H2)

img1Print = drawLines(image1Rect , warpedlines1[:10] , dst1[:10])
img2Print = drawLines(image2Rect , warpedlines1[:10] , dst1[:10])

stack2 = np.hstack((img1Print , img2Print))
cv2.imwrite(pair+"rectifiedEplines" +  ".png", stack2)


plt.imshow(stack2)
plt.show()

gray1, gray2 = gay(image1Rect) , gay(image2Rect)

disparity_scaled , disparity_unscaled = disparityMap(gray2, gray1, 10, 100)

depth_map = computeDepth(disparity_scaled , parameters[1] , k1[0][0] , parameters[7])

plt.title("disparity_unscaled")
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.imshow(disparity_scaled , cmap = 'gray')
plt.savefig(pair +' disparity_scaledGray.png')
plt.show()

plt.title("disparity heatmap")
im_color = cv2.applyColorMap(disparity_unscaled, cv2.COLORMAP_JET)
plt.imshow(cv2.cvtColor( im_color , cv2.COLOR_BGR2RGB ))
plt.savefig(pair +' disparity_mapcv2.png')
plt.show()

plt.title("depth_map depth gray")
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.imshow(depth_map , cmap = 'gray')
plt.savefig(pair +' depth_mapgray.png')
plt.show()

plt.title("depth_map cv2")
im_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
plt.imshow(im_color)
# plt.imshow(cv2.cvtColor( im_color , cv2.COLOR_BGR2RGB ))
plt.savefig(pair +' depth_mapcv2.png')
plt.show()









