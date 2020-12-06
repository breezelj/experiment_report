# import Model
# import glob
# import cv2
# import os
# import numpy as np
#
# if __name__ == '__main__':
#     # img = cv2.imread('./output/00002.bmp')
#     # kernel = np.ones((5, 5), np.uint8)
#     #
#     # img = cv2.dilate(img, kernel)
#     # img = cv2.erode(img, kernel)
#     # cv2.imshow('image', img)
#     # cv2.waitKey()
#     data_dir = 'D:/images/'
#     train_num = 200
#     gmm = Model.GMM(data_dir=data_dir, train_num=train_num)
#     gmm.train()
#     print('train finished')
#     file_list = glob.glob('D:/images/b*.bmp')
#     file_index = 0
#     for file in file_list:
#         print('infering:{}'.format(file))
#         img = cv2.imread(file)
#         img = gmm.infer(img)
#         cv2.imwrite('./output/' + '%05d' % file_index + '.bmp', img)
#         file_index += 1

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def nothing(x):
#     pass
#
#
# cv2.namedWindow('image')
#
# img = cv2.imread('./output/00002.bmp')
# cv2.namedWindow('image')
# cv2.createTrackbar('Er/Di', 'image', 0, 1, nothing)
# # 创建腐蚀或膨胀选择滚动条，只有两个值
# cv2.createTrackbar('size', 'image', 0, 21, nothing)
# # 创建卷积核大小滚动条
#
#
# while (1):
#     s = cv2.getTrackbarPos('Er/Di', 'image')
#     si = cv2.getTrackbarPos('size', 'image')
#     # 分别接收两个滚动条的数据
#     k = cv2.waitKey(1)
#
#     kernel = np.ones((si, si), np.uint8)
#     # 根据滚动条数据确定卷积核大小
#     erroding = cv2.erode(img, kernel)
#     dilation = cv2.dilate(img, kernel)
#     if k == 27:
#         break
#     # esc键退出
#     if s == 0:
#         cv2.imshow('image', erroding)
#     else:
#         cv2.imshow('image', dilation)
#         # 判断是腐蚀还是膨胀
import cv2
import numpy as np
import glob

GMM_MAX_COMPONT = 5
SIGMA = 30
gmm_thr_sumw = 0.6
train_num = 2
WEIGHT = 0.05
T = 0.5
alpha = 0.005
eps = pow(10, -10)
m_weight = [[] for i in range(GMM_MAX_COMPONT)]#怎么先对numpy初始化为一维数组，后续在对每个元素重新复制为数组呢
m_mean = [[] for i in range(GMM_MAX_COMPONT)]
m_sigma = [[] for i in range(GMM_MAX_COMPONT)]
m_fit_num = None

def init(img_gray):
    global m_fit_num
    for i in range(GMM_MAX_COMPONT):
        m_weight[i] = np.zeros_like(img_gray, dtype='float32')
        m_mean[i] = np.zeros_like(img_gray, dtype='float32')
        m_sigma[i] = np.ones(img_gray.shape, dtype='float32')
        m_sigma[i] *= SIGMA

    m_fit_num = np.zeros_like(img_gray, dtype='int32')

def train_gmm(img):
    row, col = img.shape
    m_mask = np.copy(img)
    m_mask[:] = 255
    for i in range(row):
        for j in range(col):
            num_fit = 0
            for k in range(GMM_MAX_COMPONT):
                if m_weight[k][i][j] != 0:
                    delta = abs(img[i][j]-m_mean[k][i][j])
                    if float(delta) < 2.5*m_sigma[k][i][j]:
                        m_weight[k][i][j] = (1-alpha)*m_weight[k][i][j] + alpha*1
                        m_mean[k][i][j] = (1-alpha)*m_mean[k][i][j] +alpha*img[i][j]
                        m_sigma[k][i][j] = np.sqrt((1-alpha)*m_sigma[k][i][j]*m_sigma[k][i][j]+alpha*(img[i][j]-m_mean[k][i][j])*(img[i][j]-m_mean[k][i][j]))

                        # temp = k-1
                        # for n in range(k-1, -1, -1):
                        #     temp = n
                        #     if (m_weight[n][i][j]/m_sigma[k][i][j]) <= (m_weight[n+1][i][j]/m_sigma[n+1][i][j]):
                        #         m_sigma[n][i][j], m_sigma[n+1][i][j] = m_sigma[n+1][i][j], m_sigma[n][i][j]
                        #         m_mean[n][i][j], m_mean[n+1][i][j] = m_mean[n+1][i][j], m_mean[n][i][j]
                        #         m_weight[n][i][j], m_weight[n+1][i][j] = m_weight[n+1][i][j], m_weight[n][i][j]
                        #     else:
                        #         break
                        #
                        # num_fit = temp + 1
                        # break
                        num_fit += 1
                    else:
                        m_weight[k][i][j] *= (1-alpha)


            for ii in range(GMM_MAX_COMPONT):
                for jj in range(ii+1, GMM_MAX_COMPONT):
                    if (m_weight[ii][i][j] / m_sigma[ii][i][j]) <= (m_weight[jj][i][j] / m_sigma[jj][i][j]):
                        m_sigma[ii][i][j], m_sigma[jj][i][j] = m_sigma[jj][i][j], m_sigma[ii][i][j]
                        m_weight[ii][i][j], m_weight[jj][i][j] = m_weight[jj][i][j], m_weight[ii][i][j]
                        m_mean[ii][i][j], m_mean[jj][i][j] = m_mean[jj][i][j], m_mean[ii][i][j]

            if num_fit == 0:
                if 0==m_weight[GMM_MAX_COMPONT-1][i][j]:
                    for kk in range(GMM_MAX_COMPONT):
                        if (0 == m_weight[kk][i][j]):
                            m_weight[kk][i][j] = WEIGHT
                            m_mean[kk][i][j] = img[i][j]
                            m_sigma[kk][i][j] = SIGMA
                            break
                else:
                    m_weight[GMM_MAX_COMPONT-1][i][j] = WEIGHT
                    m_mean[GMM_MAX_COMPONT-1][i][j] = img[i][j]
                    m_sigma[GMM_MAX_COMPONT-1][i][j] = SIGMA

            weight_sum = 0
            for nn in range(GMM_MAX_COMPONT):
                if m_weight[nn][i][j] != 0:
                    weight_sum += m_weight[nn][i][j]
                else:
                    break

            weight_scale = 1.0/(weight_sum)
            weight_sum = 0
            for nn in range(GMM_MAX_COMPONT):
                if m_weight[nn][i][j] != 0:
                    m_weight[nn][i][j] *= weight_scale
                    weight_sum += m_weight[nn][i][j]
                    if abs(img[i][j]-m_mean[nn][i][j]) < 2*m_sigma[nn][i][j]:
                        m_mask[i][j] = 0
                        break
                    if weight_sum > T:
                        if abs(img[i][j] - m_mean[nn][i][j]) < 2 * m_sigma[nn][i][j]:
                            m_mask[i][j] = 0
                        break
                else:
                    break

    # cv2.imshow('img_gray', img)

    # m_mask = cv2.medianBlur(m_mask, 7)
    # kernel_e = np.ones((5, 5), np.uint8)
    # m_mask = cv2.erode(m_mask, kernel_e)
    # kernel_d = np.ones((3, 3), np.uint8)
    # m_mask = cv2.dilate(m_mask, kernel_d)
    # m_mask = cv2.blur(m_mask, (5, 5))
    # m_mask = cv2.GaussianBlur(m_mask, (5, 5), 0)
    # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 形态学去噪
    # m_mask = cv2.morphologyEx(m_mask, cv2.MORPH_OPEN, element)  # 开运算去噪
    # cv2.imshow('img_mask', m_mask)
    # cv2.waitKey(1)
    return m_mask



import time

if __name__ == '__main__':
    data_dir = 'D:/images/'
    file_list = glob.glob('D:/images/b*.bmp')
    i = -1
    for file in file_list:
        # print(file)
        i += 1
        img = cv2.imread(file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if i == 0:
            init(img_gray)
        t1 = time.time()
        m_mask = train_gmm(img_gray)
        cv2.imwrite("./gray/{}.jpg".format(i), m_mask)
        t2 = time.time()
        print(t2-t1)