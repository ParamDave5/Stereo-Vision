import cv2
from cv2 import cvtColor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def gay(image):
    return cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

def printImages(images,display):
    for image in images:
        if display == True:
            plt.imshow(image)
            plt.show()

def printgrayImages(images,display):
    for image in images:
        # image = cv2.imread(i)
        if display == True:
            plt.imshow(image , cmap = 'gray')
            plt.show()

def grayImages(images):
    lst = []
    for img in images:
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        lst.append(gray)
    return np.array(lst)

def seperate(points):
    x = []
    y = []
    for i in points:
        x.append(i[0])
        y.append(i[1])
    return x,y

def sift(image ,display):
    sift = cv2.SIFT_create()
    keypoint , descriptor = sift.detectAndCompute(image , None)
    
    if display == "True":
        img = cv2.drawKeypoints(image , keypoint , image , flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(img)
        plt.title("Haaha")
        plt.show()
    return keypoint , descriptor

def featureMatching(keypoints1 , descriptor1 , keypoints2 ,descriptor2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1 , descriptor2 , k =2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    good_kp_a = []
    good_kp_b = []

    for match in good:
        good_kp_a.append(keypoints1[match.queryIdx].pt)
        good_kp_b.append(keypoints2[match.trainIdx].pt)
    return np.array(good_kp_a) , np.array(good_kp_b)

#estimate fundamental matrix using ransac we calculate the fundamental matrix and 
# compute an error xFx = 0. if the error is less than a threshold we keep that value

def fundamentalMatrix(points1 , points2):
    #create a matrix
    A = []
    for p1,p2 in zip(points1 , points2):
        x1 , y1 = p1[0] , p1[1]
        x2 , y2 = p2[0] , p1[1]
        a = [x1*x2 , x1*y2 , x1 , y1*x2 , y1*y2 , y1 , x2 , y2 , 1]
        # a = [x1*x2 , x2*y1 , x2 , y2*x1 , y2*y1 , y2 , x1 , y1 , 1]
        A.append(a)

    U , sigma , vt = np.linalg.svd(A,full_matrices=True)
    F = vt[-1:]
    F  = F.reshape(3,3)

    U ,sigma , vt = np.linalg.svd(F)
    sigma = np.diag(sigma)
    sigma[2,2] = 0
    F_ = np.dot(U , np.dot(sigma , vt))
    return F_

def calculateError(points1 , points2 , F):
    error = []
    for point1 , point2 in zip(points1 , points2):
        point1 = np.array([point1[0] , point1[1] , 1])
        point2 = np.array([point2[0] , point2[1] , 1]).T
        # err = abs(np.squeeze( np.matmul( (np.matmul(point2,F) ,point1) ) ) )
        err = abs(np.dot(points2.T , np.dot(F , points1)))
        error.append(err)
    return err

def cv2F(points1 , points2):
    F , mask = cv2.findFundamentalMat(points1 , points2 ,cv2.RANSAC, 1, 0.90)
    u , s , vt = np.linalg.svd(F)
    s[2] = 0.0
    F = u.dot(np.diag(s).dot(vt))
    return F , mask

def epipolarError(pt1 , pt2 , F):
    pt1 = np.array([pt1[0] , pt1[1] , 1])
    pt2 = np.array([pt2[0] , pt2[1] , 1]).T
    error = np.squeeze(np.dot(pt2 , np.dot(F,pt1)))
    return abs(error)

def normalize(points):
    mean = np.mean(points , axis = 0)
    xmean , ymean = mean[0] , mean[1]
    x , y = seperate(points)
    x , y = np.array(x) , np.array(y)
    xhat , yhat = x - xmean , y - ymean

    s = (2/np.mean(xhat**2 + yhat**2))**(0.5)
    T_s = np.diag([s,s,1])
    T_t = np.array([[1,0,-xmean] , [ 0,1,-ymean] , [0,0,1]])
    T = np.dot(T_s , T_t)
    x_ = np.column_stack((points , np.ones(len(points)))) 
    x_norm = (np.dot(T , x_.T)).T
    return x_norm , T

def getF(Fnorm , T2 , T1):
    Forg = T2.T @ Fnorm @ T1
    return Forg

def ransac(points1 , points2 , error_thresh , n_iter):
    points1norm , T1 = normalize(points1)
    points2norm , T2 = normalize(points2)

    max_inliers = 0
    choosen_indices = []
    choosen_f = 0
    n_rows = points1.shape[0]

    for i in range(n_iter):
        indices = []
        random_indices = np.random.choice(n_rows , size = 8)
        points1_8 = points1norm[random_indices]
        points2_8 = points2norm[random_indices]

        F = fundamentalMatrix(points1_8 , points2_8)
        # print(F)
        for j in range(n_rows):
            error = epipolarError(points1norm[j] , points2norm[j] , F)
            if error < error_thresh:
                indices.append(j)

        if len(indices) > max_inliers:
            max_inliers = len(indices)
            inliers = indices
            F_final = F
    F_ = getF(F_final , T2 , T1)
    pts1_inliers , pts2_inliers = points1[inliers] , points2[inliers]
    return F_ , pts1_inliers , pts2_inliers

def returnParams(pair):
    if pair == 'curule':
        k1 = np.array([ [1758.23 , 0 ,977.42] ,
                        [0 ,1758.23 ,552.15] ,
                        [ 0, 0 ,1] ])
        
        k2 = np.array([ [1758.23, 0 ,977.42], 
                        [0, 1758.23, 552.15] ,
                        [0, 0 ,1]  ])
        
        doffs ,baseline , width , height , ndisp , vmin , vmax , thresh = 0, 88.39, 1920 ,1080 ,220 ,55 ,195  , 10000
        parameters = [doffs ,baseline , width , height , ndisp , vmin , vmax , thresh]

    elif pair == 'octagon':
        k1=np.array([ [1742.11, 0, 804.90], 
                      [0 ,1742.11 ,541.22], 
                      [0, 0, 1] ])

        k2= np.array([ [1742.11, 0, 804.90] , 
                       [0, 1742.11, 541.22] , 
                       [0, 0, 1]])

        doffs, baseline , width , height , ndisp , vmin , vmax ,thresh =0 ,221.76 ,1920 , 1080,100,29,61 , 30000
        parameters = [doffs ,baseline , width , height , ndisp , vmin , vmax , thresh]

    elif pair == 'pendulum' :
        k1= np.array ([ [1729.05, 0, -364.24], 
                        [0, 1729.05, 552.22] , 
                        [0, 0, 1] ])

        k2= np.array ([ [1729.05, 0, -364.24] , 
                        [0, 1729.05 ,552.22], 
                        [0, 0, 1] ])

        doffs , baseline , width , height , ndisp,vmin , vmax ,thresh =0 , 537.75 , 1920,1080,180,25,150 , 150000
        parameters = [doffs ,baseline , width , height , ndisp , vmin , vmax , thresh]

    return k1 , k2 , parameters

def getEssentialMatrix(k1 , k2,F):
    E = k2.T.dot(F).dot(k1)
    u , s , v = np.linalg.svd(E)
    s = [1,1,0]
    E_final = np.dot(u , np.dot(np.diag(s),v))
    return E_final

def decomposeE(E):
    u,s,vt = np.linalg.svd(E)
    W = np.array([[0,-1,0] , [1,0,0] , [0,0,1]])
    c1 = u[:,2]
    r1 = np.dot(u , np.dot(W , vt))
    c2 = -u[:,2]
    r2 = np.dot(u , np.dot(W , vt))
    c3 = u[:,2]
    r3 = np.dot(u , np.dot(W.T , vt))
    c4 = -u[:,2]
    r4 = np.dot(u , np.dot(W.T , vt))

    R = np.array([r1,r2,r3,r4] , dtype =np.float32)
    C = np.array([c1,c2,c3,c4] , dtype = np.float32)
    for i in range(4):
        if np.linalg.det(R[i]) < 0:
            R[i] = -R[i]
            C[i] = -C[i]
    return R , C

def projectionMatrix(K , R , C):
    I = np.identity(3)
    P = np.dot(K , np.dot(R , np.hstack((I , -C))))
    return P

def AMatrix(pt1,pt2,P1,P2):
    p1 , p2 , p3 = P1
    p11 , p12 , p13 = P2
    p1 , p2 , p3 = p1.reshape(1,-1) , p2.reshape(1,-1) ,p3.reshape(1,-1) 
    p11 , p12 , p13 = p11.reshape(1,-1) , p12.reshape(1,-1) ,p13.reshape(1,-1) 
    x,y = pt1
    x1 , y1 = pt2
    A = np.vstack((y*p3 - p2 , p1 - x*p3  , 
                    y1*p13 - p12 , p11 - x1*p13))
    return A

def triangulation(points1 , points2 , R1 , R2 , C1 , C2 , k1 , k2):
    P1 = projectionMatrix(k1 , R1 , C1)
    P2 = projectionMatrix(k2 , R2 , C2)
    point3d = []
    for p1,p2 in zip(points1 , points2):
        A = AMatrix(p1 , p2 , P1 , P2)
        u , s , vt = np.linalg.svd(A)
        x = vt[-1]
        x = x/x[-1]
        point3d.append(x[:3])
    return point3d

def linearTriangulation(Rset , Cset , points1 , points2 , k1,k2):
    points3d = []
    for i in range(len(Rset)):
        R1 , R2 = np.identity(3) , Rset[i]
        C1 , C2 = np.zeros((3,1)) , Cset[i].reshape(3,1)
        point3d = triangulation(points1 , points2 , R1 , R2 , C1 , C2 , k1,k2)
        points3d.append(point3d)
    return points3d

def depthPositivityConstraint(pts , r3 , C):
    n = 0
    for X in pts:
        X = X.reshape(-1,1)
        if np.dot(r3 , (X-C)) > 0  and X[2]>0:  
            n += 1
    return n

def recoverPose(E , points1 , points2 , k1 , k2):
    #get3d points using linear triangulation method
    Rset , Cset = decomposeE(E)
    points3d = linearTriangulation(Rset , Cset , points1 , points2 , k1,k2)
    best = 0
    max_depth = 0
    for i in range(len(Rset)):
        R , C = Rset[i] , Cset[i].reshape(-1,1)
        r3 = R[2].reshape(1,-1)
        pt3d = points3d[i]
        n = depthPositivityConstraint(pt3d , r3 , C)
        if n >  max_depth:
            best = i
            max_depth = n
        
    R , C , pts3d = Rset[best] , Cset[best] , points3d[best]
    return R , C , pts3d

def drawEpilines(image1 , image2 , lines , pts1 , pts2):
    img1 , img2 = image1.copy() , image2.copy()
    lines = lines.reshape(-1,3)
    # img1 = np.array(img1)
    r,c = image1.shape[:2]
    for r,pt1 , pt2 in zip(lines ,np.int32(pts1) , np.int32(pts2)):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int , [0, -r[2]/r[1]])
        x1 , y1 = map(int , [c, -(r[2]+r[0]*c)/r[1]] )

        img1 = cv2.line(img1 , (x0,y0) , (x1,y1) , color ,1)
        img1 = cv2.circle(img1 , tuple(pt1) , 2 , color ,-1)
        img2 = cv2.circle(img2 , tuple(pt2) , 2 , color ,-1)
    return img1 , img2

def epilines(pts1 , image , F , width):
    lines = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2),image ,F ).reshape(-1,3)
    linesOut = []
    for r in lines:
        x , y = map(int , [0 , -r[2]/r[1]])
        x1,y1 = map(int ,[width , -(r[2]+r[0]*width)/r[1] ] )
        epiline = [[x,y] , [x1,y1]]
        linesOut.append(epiline)
    return np.array(linesOut)

def warpEpilines(lines , H):
    start = np.float32(lines[:,0].reshape(-1,1,2))
    end = np.float32(lines[:,1].reshape(-1,1,2))

    start_ = cv2.perspectiveTransform(start , H).squeeze()
    end_ = cv2.perspectiveTransform(end , H).squeeze()

    lines = []
    for start ,end in zip(start_ , end_):
        lines.append([start,end])
    lines = np.array(lines)
    return lines

def drawLines(image , lines ,pts):
    img1 =image.copy()
    for l , pt in zip(lines , pts):
        l_color = (0,255,0)
        (x0,y0) , (x1,y1) = l[0] , l[1]
        img1  = cv2.line(img1 , (int(x0),int(y0)) , (int(x1),int(y1)) , l_color , 1)
        p_color = (0,0,255)
        x,y = pt[0] , pt[1]
        cv2.circle(img1 , (int(x) ,int(y)) , 1 , p_color ,-1)
    return img1

#compute ssd
#for each point in the left image calculate the disparity width - xi

def SSD(win1 , win2):
    if win1.shape != win2.shape:
        return -1
    ssd = np.sum(np.square(win1 - win2))
    return ssd

def disparityMap(gray1, gray2, window_size, search_range):
    height , width = gray1.shape
   
    disparity_map = np.zeros_like(gray1)
    gray1 , gray2 = gray1.astype(np.float64) , gray2.astype(np.float64)
    for y in tqdm(range(window_size, height-window_size)):
        for x in range(window_size, width - window_size):
            window = gray1[y-(window_size//2):y+(window_size//2) , x-(window_size//2):x+(window_size//2)]
            x1 = blockMatching(y,x,window,gray2 , window_size , 56)
            disparity_map[y,x] = (x1 - x)
    # disparity_map_unscaled = disparity_map.copy()
    # disparity_map_scaled =  disparity_map.copy()
    max_pixel = np.max(disparity_map)
    min_pixel = np.min(disparity_map)
    # for i in range(disparity_map.shape[0]):
    #     for j in range(disparity_map.shape[1]):
    #         disparity_map[i][j] = int((disparity_map[i][j]*255))/(max_pixel - min_pixel)

    # disparity_map_unscaled = disparity_map.copy()
    disparity_map_scaled = disparity_map + np.abs(min_pixel)
    disparity_map_unscaled = (disparity_map_scaled/np.max(disparity_map_scaled)*255)
    # disparity_map_scaled =  disparity_map.copy()
    return disparity_map_scaled , disparity_map_unscaled.astype(np.uint8)

def blockMatching(y,x,window,gray2 , window_size , searchRange):
    height1 , width1 = gray2.shape
    x_start = max(0, x - searchRange)
    x_end = min(width1 , x + searchRange)
    min_x = np.inf
    min_ssd  = np.inf
    
    for x in range(x_start , x_end,window_size):
        window2 = gray2[y-(window_size//2):y+(window_size//2) , x-(window_size//2):x+(window_size//2)]
        # print(window.shape , window2.shape)
        ssd = SSD(window , window2)
        if ssd < min_ssd:
            min_ssd = ssd 
            min_x = x
    return  min_x

def computeDepth(disparity_map , baseline , f ,thresh):
    # depth_map = np.zeros((disparity_map.shape[0] , disparity_map.shape[0]))
    depth_map = (baseline*f)/(disparity_map + 1e-10)
    depth_map[depth_map > thresh] = thresh
    depth_map = np.uint8(depth_map *255 / np.max(depth_map))
    return depth_map























 




