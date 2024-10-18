#! /usr/bin/env python


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2

#prinit information about library
print("OpenCV version:", cv2.__version__)
build_info = cv2.getBuildInformation()
print("OpenCV build information:")
print(build_info)



#read file
image = cv2.imread("mandrill.png")
print("Shape = ", image.shape)

print("Image as an array: ")
print(image[0:5,0:5])


#split colors:
blue, green, red = cv2.split(image)
print("Red channel: \n", red)

#show plots of r, g, b channels
x = np.linspace(0, image.shape[0], image.shape[0])
y = np.linspace(0, image.shape[1], image.shape[1])
X, Y = np.meshgrid(x, y)
plt.pcolor(X, Y, red)
plt.savefig("output/Red.png")
plt.show()

plt.pcolor(X, Y, green)
plt.savefig("output/Green.png")
plt.show()

plt.pcolor(X, Y, blue)
plt.savefig("output/Blue.png")
plt.show()


for name, color in zip(("red", "green", "blue"), (red, green, blue)):
    ax = plt.figure().add_subplot(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, color, linewidth=0, antialiased=False)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel(name)
    plt.savefig("output/3D{}.png".format(name))
    plt.show()


#join channels
img = cv2.merge([blue, green, red])
print("same :", (img == image).all())


#convert to HSV
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#convert to gray
imageGrey =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


##write to file
cv2.imwrite('output/MandrillBW.png', imageGrey)


#window preview
cv2.imshow('Image', image) 
  
##wait for user to press any key 
cv2.waitKey(0) 
  
## destroy all created windows
cv2.destroyAllWindows() 


# crop image
cropped = image[20:100, 130:230, :]
cv2.imwrite("output/cropped.png", cropped)




