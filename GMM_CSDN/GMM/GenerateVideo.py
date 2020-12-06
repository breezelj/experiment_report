import cv2 as cv
import numpy as np
import glob as glob

left_list = sorted(glob.glob('D:/images/b*.bmp'))
right_list = sorted(glob.glob('./rgb_mb7_dila5_train200/*.jpg'))
VW = cv.VideoWriter('./video/test.avi', cv.VideoWriter_fourcc(*'MJPG'), 25, (320, 120))
for i in range(len(left_list)):
    print(left_list[i])
    print(right_list[i])
    result = np.concatenate((cv.imread(left_list[i]), cv.imread(right_list[i])), axis=1)
    cv.imwrite('./video/' + '%05d' % i + '.bmp', result)
    VW.write(result)
