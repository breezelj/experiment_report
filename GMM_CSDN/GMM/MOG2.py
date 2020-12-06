import cv2
import glob

fgbg = cv2.createBackgroundSubtractorMOG2()#混合高斯背景建模算法

img_list = sorted(glob.glob('D:/WavingTrees/*.bmp'))
for i in range(len(img_list)):
    frame = cv2.imread(img_list[i])

    fgmask = fgbg.apply(frame)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))  # 形态学去噪
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)  # 开运算去噪

    cv2.imwrite("./MOG2/{}.jpg".format(i), fgmask)
