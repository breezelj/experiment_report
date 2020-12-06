import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
x_list, y_list = [], []

def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        print(x, y)
        x_list.append(x)
        y_list.append(y)



img_list = sorted(glob.glob('D:/WavingTrees/*.bmp'))
cv2.namedWindow("image")

pix_list = []
for i in range(len(img_list)):
    img = cv2.imread(img_list[i])
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if i == 0:
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
        cv2.imshow("image", img)
        cv2.waitKey(0)
    if i == 0:
        cv2.circle(img, (x_list[0], y_list[0]), 1, (0, 0, 255), 4)
        cv2.imshow('pix_img', img)
        cv2.imwrite('mark_img.jpg', img)
        cv2.waitKey(0)
    pix_list.append(img_gray[x_list[0], y_list[0]])
    # cv2.imshow("image", img)
    # cv2.waitKey(1)
print(min(pix_list), max(pix_list))
plt.scatter(list(np.arange(len(pix_list))), pix_list)
plt.xlabel('frames')
plt.ylabel('pixel value')
plt.savefig('pixel_distribution.jpg')
plt.show()


