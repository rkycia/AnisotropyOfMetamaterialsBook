#! /usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2


img = cv2.imread("Picture2.png")

#cut out printout
img = img[50:-80,10:]
cv2.imwrite("output/Picture2Cut.png", img)


#histograms of colors
hist= cv2.calcHist([img],[0],None,[256],[0,256])
hist2=cv2.calcHist([img],[1],None,[256],[0,256])
hist3=cv2.calcHist([img],[2],None,[256],[0,256])
##plot histograms
plt.subplot(1, 3, 1)
plt.plot(hist/max(hist))   
plt.title('Hist Blue : ')
plt.subplot(1, 3, 2)
plt.plot(hist2/max(hist2),color="green")   
plt.title('Hist Green : ')
plt.subplot(1, 3, 3)
plt.plot(hist3/max(hist3),color="red")   
plt.title('Hist Red : ')
plt.savefig("output/Picture2ColorHistograms")
plt.show()



#histogram equalization
src = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(src)
v = cv2.equalizeHist(v)
src = cv2.merge([h,s,v])
img1 = cv2.cvtColor(src, cv2.COLOR_HSV2BGR)
cv2.imwrite("output/Picture2Equalization.png", img1)


#histograms of colors after normalization
hist= cv2.calcHist([img1],[0],None,[256],[0,256])
hist2=cv2.calcHist([img1],[1],None,[256],[0,256])
hist3=cv2.calcHist([img1],[2],None,[256],[0,256])
plt.subplot(1, 3, 1)
plt.plot(hist/max(hist))   
plt.title('Hist Blue : ')
plt.subplot(1, 3, 2)
plt.plot(hist2/max(hist2),color="green")   
plt.title('Hist Green : ')
plt.subplot(1, 3, 3)
plt.plot(hist3/max(hist3),color="red")   
plt.title('Hist Red : ')
plt.savefig("output/Picture2ColorHistogramsEqualization.png")
plt.show()




#scale figure
img1 = cv2.resize(img1,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
img = cv2.resize(img,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
# convert to BW
img1_BW = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/Picture2BW.png", img1_BW)


#bilateral filtering
img1_BW = cv2.bilateralFilter(img1_BW,9,75,75)
cv2.imwrite("output/Picture2BWFiltering.png", img1_BW)


#compute gray histogram
hist1 = cv2.calcHist([img1_BW], [0], None, [256], [0,256])
plt.plot(hist1.ravel()/hist1.max())
plt.savefig("output/Picture2HistogramBW.png")
plt.show()

#tresholding - we select 40 for selecting black granules
ret1m, thresh1 = cv2.threshold(img1_BW, 50, 256, cv2.THRESH_BINARY)

#remove small dots
thresh1 = cv2.dilate(thresh1, np.ones((5, 5), np.uint8))
thresh1 = cv2.erode(thresh1, np.ones((5, 5), np.uint8))

cv2.imwrite("output/Picture2BWThresholding.png", thresh1)

#reverse colors in mask
thresh2 = abs(thresh1 - 255)


#find contours
contours, _ = cv2.findContours(thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1 )
blank = np.ones(thresh2.shape)*255
cv2.drawContours(blank, contours, -1, (0, 0, 0), 3)
print("# of contours found: ", len(contours))
cv2.imwrite("output/Picture2BWCountour.png", blank)




#minmimal area rectangle
img1_tmp1 = img.copy()
img1_tmp2 = img.copy()
for idx, cnt in enumerate(contours):
    minRect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(minRect)
    box = np.int0(box)
    cv2.drawContours(img1_tmp1, [box], 0, (0,0,255), 2)
    cv2.drawContours(img1_tmp2, [box], 0, (0,0,255), 2)
    cv2.putText(img1_tmp2, '{}'.format(idx), (int(box[0][0]), int(box[0][1]) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 
cv2.imwrite("output/Picture2BWMinAreaRectangle.png", img1_tmp1)
cv2.imwrite("output/Picture2BWMinAreaRectangleWithIndices.png", img1_tmp2)


#collect images
#image = img1.copy()
image = img.copy()
img_tmp = img1.copy()
filtered_contours = []
selected_img = []
for idx, contour in enumerate(contours):
    #draw selected contur
    minRect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(minRect)
    box = np.int0(box)
    cv2.drawContours(img_tmp, [box], -1, (0,0,255), 1)
        
    #extract image from bouding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    #cropped = image[y:y + h, x:x + w, :].copy()
    a = 20
    cropped = image[y-a if y-a >=0 else 0 :y + h+a, x-a if x-a >= 0 else 0:x + w+a, :].copy()
    feature = cv2.resize(cropped, (40, 40))
    cv2.imwrite("output/figs/{}.png".format(idx), feature)
    
    #scale cropped image to 1D array
    #feature = feature.reshape(-1)
    selected_img.append(feature)
    
cv2.imwrite("output/Picture2SelectedContours.png", img_tmp)

selected_img = np.array(selected_img)

print("selected_img = ", selected_img.shape)


#load NN model
import tensorflow as tf

model = tf.keras.models.load_model('model/NNModel.h5')

#classify subimages
y = model.predict(selected_img)

#print((y>0.5).any())

#quantize output to 0 or 1
y = np.where(y>0.5, 1, 0)

print("number of cancer cells: ", np.sum(y))

#img_tmp = image.copy()
#img_ellipse = image.copy()
#img_ellipse_closed = image.copy()

img_tmp = img1.copy()
img_ellipse = img1.copy()
img_ellipse_closed = img1.copy()

ellipsesData = []
for idx, contour in enumerate(contours):
   
    minRect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(minRect)
    box = np.int0(box)
    cv2.drawContours(img_tmp, contour, -1, (50,50,50), 1)
    

        
    #fit ellipses
    if contour.size < 10 :
        continue
    ellipse = cv2.fitEllipse(contour)
    if max(ellipse[1]) < 150:
        label = y[idx][0]
        #print("label = ", label)
        ellipsesData.append({"id": idx, "x":ellipse[0][0], "y":ellipse[0][1], "majorAxis": ellipse[1][0], "minorAxis": ellipse[1][1], "angle": ellipse[2], "class": label})
        #cv2.ellipse(img_ellipse, ellipse, (0,0,label*255), 1, cv2.LINE_AA)
        
        cv2.putText(img_ellipse, '{}'.format(label), (int(box[0][0]), int(box[0][1]) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 
        #cv2.ellipse(img_ellipse_closed, ellipse, (0,0,label*250), -1, cv2.LINE_AA)
        cv2.putText(img_ellipse_closed, '{}'.format(label), (int(box[0][0]), int(box[0][1]) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 
        cv2.putText(img_tmp, '{}'.format(label), (int(box[0][0]), int(box[0][1]) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 
        if label == 1:
            cv2.drawContours(img_tmp, [box], -1, (0,0,250), 1)
            #cv2.drawContours(img_ellipse, [box], -1, (0,0,250), 1)
            #cv2.drawContours(img_ellipse_closed, [box], -1, (0,0,250), 1)
            cv2.ellipse(img_ellipse, ellipse, (0,0,255), 1, cv2.LINE_AA)
            cv2.ellipse(img_ellipse_closed, ellipse, (0,0,250), -1, cv2.LINE_AA)
        else:
            cv2.drawContours(img_tmp, [box], -1, (0,250,0), 1)
            #cv2.drawContours(img_ellipse, [box], -1, (0,250,0), 1)
            #cv2.drawContours(img_ellipse_closed, [box], -1, (0,250,0), 1)
            cv2.ellipse(img_ellipse, ellipse, (0,255,0), 1, cv2.LINE_AA)
            cv2.ellipse(img_ellipse_closed, ellipse, (0,250,0), -1, cv2.LINE_AA)

df_ellipses = pd.DataFrame.from_records(ellipsesData)
df_ellipses.to_csv("output/ellipses.csv")

cv2.imwrite('output/Picture2_color_segmentation.png', img_tmp)
cv2.imwrite('output/Picture2_color_ellypses.png', img_ellipse)
cv2.imwrite('output/Picture2_color_ellypses_closed.png', img_ellipse_closed)

#angle histogram
plt.hist(np.array(df_ellipses.angle)%90)
plt.xlabel("angle[deg]  mod 90")
plt.ylabel("No. of ellipses")
plt.savefig("output/Picture2AngleDistibution.png")
plt.show()

