# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 19:58:24 2016

@author: lvye
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
image = mpimg.imread('../test_images/solidWhiteRight.jpg')
import math


plt.imshow(image)
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:         
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., lambd=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lambd)

def colorSelection(img,colorRange):    
    mask = cv2.inRange(img, colorRange[0], colorRange[1])
    return mask

def regionSelection(img,Vertices):
    blank = np.zeros(img.shape,np.uint8)
    color = np.array([255,255,255])
    cv2.fillPoly(blank, [Vertices], color)
    return blank

def separateLRLines(lines,width,height):
    leftlines = []
    rightlines = []
    left_k = []
    right_k = []
    for i in range(len(lines)):
        line = lines[i][0]
        #print line
        x0 = line[0]
        y0 = line[1]
        dx = line[0]-line[2]
        dy = line[1]-line[3]
        if dy==0:
            continue
        #line extends to bottom
        x_ = 1.0*dx/dy*(height-y0)+x0
        if(x_<2*width/5):
            leftlines.append([line])
            left_k.append(1.0*dx/dy)
        elif(x_>3*width/5):
            rightlines.append([line])
            right_k.append(1.0*dx/dy)
    return left_k,right_k,leftlines,rightlines

def findGoodLeftandRightLine(left_k,right_k,leftLines,rightLines,width,height):
    n_left = len(leftLines)
    n_right = len(rightLines)    
    min_l_y=99999999.0
    min_r_y=99999999.0
    l_y=-1.0;l_x=-1.0;
    r_y=-1.0;r_x=-1.0;
    for i in range(n_left):
        L_line = leftLines[i][0]        
        for j in range(2):
            x1 = L_line[0+2*j]
            y1 = L_line[1+2*j]
            if y1<min_l_y:
                print y1
                min_l_y = y1
                l_y = y1
                l_x = x1
    
    for j in range(n_right):
        R_line = rightLines[j][0]    
        for j in range(2):
            x2 = R_line[0+2*j]
            y2 = R_line[1+2*j]
            if y2<min_r_y:
                #print y2
                min_r_y = y2
                r_y = y2
                r_x = x2    
                
    leftLine=[]
    if n_left>0:        
        k_l=sum(left_k)/n_left
        l_bottom_x = k_l*(height-l_y)+l_x    
        leftLine = [int(l_x),int(l_y),int(l_bottom_x),int(height)]  
        
    rightLine = []           
    if n_right>0:        
        k_r=sum(right_k)/n_right
        r_bottom_x = k_r*(height-r_y)+r_x
        rightLine = [int(r_x),int(r_y),int(r_bottom_x),int(height)]
            
    return leftLine,rightLine
    
def RoadLaneDetectionPipeLine(image,showIntemediateResult=False,showFinalResult=False):   
    ###############################################################
    #houghLine detection
    #set parameter
    ###############################################################
    rho_resolution = 1
    theta_resolution = np.pi/360
    threshold = 20
    min_line_len = 100
    max_line_gap = 500
    ###############################################################
    img = grayscale(image)
    height = img.shape[0]
    width = img.shape[1]
    ###############################################################
    #specify region mask vertices
    Vertices = np.array([[0,height],[width,height],[width/2,int(1.0*height/2.0)]])
    #specify color range to be accepted
    colorRange = np.array([200,255])#lower and higher   
    ###############################################################    
    
    if showIntemediateResult:
        plt.imshow(img, cmap='gray')    
     
    #get color mask and region mask
    colorMask = colorSelection(img,colorRange)
    regionMask = regionSelection(img,Vertices)
    #show masks
    if showIntemediateResult:
        plt.title("colorMask")
        plt.imshow(colorMask, cmap='gray')
        plt.figure()
        plt.title("regionMask")
        plt.imshow(regionMask, cmap='gray')
        plt.figure()

    #filterd image by color mask
    colorFilteredImg = cv2.bitwise_and(img,img,mask = colorMask)
    if showIntemediateResult:
        plt.figure()
        plt.title("colorFilteredImg")
        plt.imshow(colorFilteredImg, cmap='gray')

    #canny on image filtered by color mask
    low_threshold = 50
    high_threshold = 150
    edges = canny(colorFilteredImg, low_threshold, high_threshold)
    if showIntemediateResult:
        plt.figure()
        plt.title("edges")
        plt.imshow(edges, cmap='gray')

    #apply region mask on edges
    edgesFilteredbyRegionMask = cv2.bitwise_and(edges,edges,mask = regionMask)
    if showIntemediateResult:
        plt.figure()
        plt.title("edgesFilteredbyRegionMask")
        plt.imshow(edgesFilteredbyRegionMask, cmap='gray')
    
    if showIntemediateResult:
        lineImg = hough_lines(edgesFilteredbyRegionMask, rho_resolution, theta_resolution, threshold, min_line_len, max_line_gap)

        plt.figure()
        plt.title("hough_lines")
        plt.imshow(lineImg)

        #addweighted
        outputImg = cv2.addWeighted(lineImg,1,image,1,0)
        #if showIntemediateResult:
        plt.figure()
        plt.title("hough_lines_image")
        plt.imshow(outputImg)        
        
    #find left right Lines 
    lines = cv2.HoughLinesP(edgesFilteredbyRegionMask, rho_resolution, theta_resolution, threshold, np.array([]),\
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    left_k,right_k,leftlines,rightlines = separateLRLines(lines,width,height)
    leftLine,rightLine = findGoodLeftandRightLine(left_k,right_k,leftlines,rightlines,width,height)
    Lines=[]
    if leftLine:
        Lines.append([leftLine])
    if rightLine:
        Lines.append([rightLine])    
    line_img = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
    draw_lines(line_img, Lines,thickness=10)
    if showIntemediateResult:
        plt.figure()
        plt.title("hough_extent_lines")
        plt.imshow(line_img)
    #addweighted
    outputImg = cv2.addWeighted(line_img,1,image,1,0)
    if showFinalResult:
        plt.figure()
        plt.title("hough_extent_lines_image")
        plt.imshow(outputImg)

    return outputImg

outPut = RoadLaneDetectionPipeLine(image,showIntemediateResult=True,showFinalResult=True)