#! /usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2


img1 = cv2.imread("Picture2.png")



#cut out printout
img1 = img1[50:-80,10:]
cv2.imwrite("output/Example4Cut.png", img1)


#histograms of colors
hist= cv2.calcHist([img1],[0],None,[256],[0,256])
hist2=cv2.calcHist([img1],[1],None,[256],[0,256])
hist3=cv2.calcHist([img1],[2],None,[256],[0,256])
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
plt.savefig("output/Example4ColorHistograms")
plt.show()



#histogram equalization
src = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(src)
v = cv2.equalizeHist(v)
src = cv2.merge([h,s,v])
img1 = cv2.cvtColor(src, cv2.COLOR_HSV2BGR)
cv2.imwrite("output/Example4Equalization.png", img1)


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
plt.savefig("output/Example4ColorHistogramsEqualization.png")
plt.show()




#scale figure
img1 = cv2.resize(img1,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
# convert to BW
img1_BW = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/Example4BW.png", img1_BW)


#bilateral filtering
img1_BW = cv2.bilateralFilter(img1_BW,9,75,75)
cv2.imwrite("output/Example4BWFiltering.png", img1_BW)


#compute gray histogram
hist1 = cv2.calcHist([img1_BW], [0], None, [256], [0,256])
plt.plot(hist1.ravel()/hist1.max())
plt.savefig("output/Example4HistogramBW.png")
plt.show()

#tresholding - we select 40 for selecting black granules
ret1m, thresh1 = cv2.threshold(img1_BW, 50, 256, cv2.THRESH_BINARY)

#remove small dots
thresh1 = cv2.dilate(thresh1, np.ones((5, 5), np.uint8))
thresh1 = cv2.erode(thresh1, np.ones((5, 5), np.uint8))

cv2.imwrite("output/Example4BWThresholding.png", thresh1)

#reverse colors in mask
thresh2 = abs(thresh1 - 255)


#find contours
contours, _ = cv2.findContours(thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1 )
blank = np.ones(thresh2.shape)*255
cv2.drawContours(blank, contours, -1, (0, 0, 0), 3)
print("# of contours found: ", len(contours))
cv2.imwrite("output/Example4BWCountour.png", blank)




#minmimal area rectangle
img1_tmp1 = img1.copy()
img1_tmp2 = img1.copy()
for idx, cnt in enumerate(contours):
    minRect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(minRect)
    box = np.int0(box)
    cv2.drawContours(img1_tmp1, [box], 0, (0,0,255), 2)
    cv2.drawContours(img1_tmp2, [box], 0, (0,0,255), 2)
    cv2.putText(img1_tmp2, '{}'.format(idx), (int(box[0][0]), int(box[0][1]) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 
cv2.imwrite("output/Example4BWMinAreaRectangle.png", img1_tmp1)
cv2.imwrite("output/Example4BWMinAreaRectangleWithIndices.png", img1_tmp2)


#collect data
import math
image = img1.copy()
img_tmp = img1.copy()
filtered_contours = []
selected_data = []
for idx, contour in enumerate(contours):
    # get mean color of contour:
    masked = np.zeros_like(image[:, :, 0])
    cv2.drawContours(masked, [contour], 0, 255, -1)

    B_mean, G_mean, R_mean, _ = cv2.mean(image, mask=masked)
    
    #calculate moments
    moments = cv2.moments(contour)
    # Calculate Hu Moments 
    huMoments = cv2.HuMoments(moments)

    #scaling Hu moments
    for i in range(0,7):
        #print("hu moment ", i, " ", huMoments[i])
        if huMoments[i] == 0:
            huMoments[i] = 0.0
        else:
            huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
    
    selected_data.append({'B_mean': B_mean, 'G_mean': G_mean, 'R_mean': R_mean, 'h0':huMoments[0], 'h1':huMoments[1], 'h2':huMoments[2], 'h3':huMoments[3], 'h4':huMoments[4], 'h5':huMoments[5], 'h6':huMoments[6]})

    
cv2.imwrite("output/Exmaple2SelectedContours.png", img_tmp)

df = pd.DataFrame(selected_data)

#k-means number of clusters estimation
from sklearn.cluster import KMeans

k = range(1,20)
WSSE = []

for i in k:
    model = KMeans(n_clusters = i, n_init='auto')
    model.fit_predict(df)
    WSSE.append(model.inertia_)

plt.plot(k,WSSE)
plt.xlabel('K')
plt.ylabel('WSSE')
plt.grid(True)
plt.savefig("output/Example4kmeans.png")
plt.show()


#k-means
km = KMeans( n_clusters=4, n_init='auto')
df['label'] = km.fit_predict(df)

df.to_csv("output/LabelledContours.csv")

img_tmp = image.copy()
img_ellipse = image.copy()
img_ellipse_closed = image.copy()
ellipsesData = []
for label, df_grouped in df.groupby('label'):
    Ngranules = len(df_grouped)
    print(Ngranules, " in label ", label)
    mask = np.zeros_like(image[:, :, 0])
    cv2.drawContours(mask, [contours[i] for i in df_grouped.index], -1, (255), -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    masked_image[mask==0] = (255,255,255)
    cv2.imwrite("output/Example4label{}.png".format(label), masked_image)
    
    print("fitting for label = ", label)
    for  idx in df_grouped.index:
        minRect = cv2.minAreaRect(contours[idx])
        box = cv2.boxPoints(minRect)
        box = np.int0(box)
        cv2.drawContours(img_tmp, contours[idx], -1, (0,0,label*50), 1)
        
        #fit ellypses
        if contours[idx].size < 10 :
            continue
        ellipse = cv2.fitEllipse(contours[idx])
        ellipsesData.append({"id": idx, "x":ellipse[0][0], "y":ellipse[0][1], "majorAxis": ellipse[1][0], "minorAxis": ellipse[1][1], "angle": ellipse[2], "class": label})
        if max(ellipse[1]) < 150:
            cv2.ellipse(img_ellipse, ellipse, (0,0,label*50), 1, cv2.LINE_AA)
            cv2.putText(img_ellipse, '{}_{}'.format(idx, label), (int(box[0][0]), int(box[0][1]) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 
            cv2.ellipse(img_ellipse_closed, ellipse, (0,0,label*50), -1, cv2.LINE_AA)
            cv2.putText(img_ellipse_closed, '{}_{}'.format(idx, label), (int(box[0][0]), int(box[0][1]) ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 

df_ellipses = pd.DataFrame.from_records(ellipsesData)
df_ellipses.to_csv("output/ellipses.csv")

cv2.imwrite('output/Example4_color_segmentation.png', img_tmp)
cv2.imwrite('output/Example4_color_ellypses.png', img_ellipse)
cv2.imwrite('output/Example4_color_ellypses_closed.png', img_ellipse_closed)

#angle histogram
plt.hist(np.array(df_ellipses.angle)%90)
plt.xlabel("angle[deg]  mod 90")
plt.ylabel("No. of ellipses")
plt.savefig("output/Example4AngleDistibution.png")
plt.show()

