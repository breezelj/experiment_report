import cv2
import numpy as np
import glob

GMM_MAX_COMPONT = 5
SIGMA = 30
gmm_thr_sumw = 0.6
train_num = 2
WEIGHT = 0.05
T = 0.7
alpha = 0.005
eps = pow(10, -10)
channel = 3
GMM_MAX_COMPONT = GMM_MAX_COMPONT
m_weight = [[] for i in range(GMM_MAX_COMPONT*channel)]#怎么先对numpy初始化为一维数组，后续在对每个元素重新复制为数组呢
m_mean = [[] for i in range(GMM_MAX_COMPONT*channel)]
m_sigma = [[] for i in range(GMM_MAX_COMPONT*channel)]
m_fit_num = None

def init(img):
    row, col, channel = img.shape
    global m_fit_num
    for i in range(GMM_MAX_COMPONT*channel):
        m_weight[i] = np.zeros((row, col), dtype='float32')
        m_mean[i] = np.zeros((row, col), dtype='float32')
        m_sigma[i] = np.ones((row, col), dtype='float32')
        m_sigma[i] *= SIGMA

    m_fit_num = np.zeros((row, col), dtype='int32')

def train_gmm(imgs):
    row, col, channel = imgs.shape
    B, G, R = cv2.split(imgs)
    m_mask = np.zeros((row,col), dtype=np.uint8)
    m_mask[:] = 255
    for i in range(row):
        for j in range(col):
            cnt = 0
            for c, img in enumerate((B,G,R)):
                # img = imgs[:,:,c]
                # img = B.copy()
                num_fit = 0
                for k in range(c*GMM_MAX_COMPONT, c*GMM_MAX_COMPONT+GMM_MAX_COMPONT):
                    if m_weight[k][i][j] != 0:
                        delta = abs(img[i][j]-m_mean[k][i][j])
                        if float(delta) < 2.5*m_sigma[k][i][j]:
                            m_weight[k][i][j] = (1-alpha)*m_weight[k][i][j] + alpha*1
                            m_mean[k][i][j] = (1-alpha)*m_mean[k][i][j] +alpha*img[i][j]
                            m_sigma[k][i][j] = np.sqrt((1-alpha)*m_sigma[k][i][j]*m_sigma[k][i][j]+alpha*(img[i][j]-m_mean[k][i][j])*(img[i][j]-m_mean[k][i][j]))

                            num_fit += 1

                        else:
                            m_weight[k][i][j] *= (1-alpha)

                for ii in range(c*GMM_MAX_COMPONT, c*GMM_MAX_COMPONT+GMM_MAX_COMPONT):
                    for jj in range(ii + 1, c*GMM_MAX_COMPONT+GMM_MAX_COMPONT):
                        if (m_weight[ii][i][j] / m_sigma[ii][i][j]) <= (m_weight[jj][i][j] / m_sigma[jj][i][j]):
                            m_sigma[ii][i][j], m_sigma[jj][i][j] = m_sigma[jj][i][j], m_sigma[ii][i][j]
                            m_weight[ii][i][j], m_weight[jj][i][j] = m_weight[jj][i][j], m_weight[ii][i][j]
                            m_mean[ii][i][j], m_mean[jj][i][j] = m_mean[jj][i][j], m_mean[ii][i][j]

                if num_fit == 0:
                    if 0==m_weight[c*GMM_MAX_COMPONT+GMM_MAX_COMPONT-1][i][j]:
                        for kk in range(c*GMM_MAX_COMPONT, c*GMM_MAX_COMPONT+GMM_MAX_COMPONT):
                            if (0 == m_weight[kk][i][j]):
                                m_weight[kk][i][j] = WEIGHT
                                m_mean[kk][i][j] = img[i][j]
                                m_sigma[kk][i][j] = SIGMA
                                break
                    else:
                        m_weight[c*GMM_MAX_COMPONT+GMM_MAX_COMPONT-1][i][j] = WEIGHT
                        m_mean[c*GMM_MAX_COMPONT+GMM_MAX_COMPONT-1][i][j] = img[i][j]
                        m_sigma[c*GMM_MAX_COMPONT+GMM_MAX_COMPONT-1][i][j] = SIGMA

                weight_sum = 0
                for nn in range(c*GMM_MAX_COMPONT, c*GMM_MAX_COMPONT+GMM_MAX_COMPONT):
                    if m_weight[nn][i][j] != 0:
                        weight_sum += m_weight[nn][i][j]
                    else:
                        break

                weight_scale = 1.0/(weight_sum+eps)
                weight_sum = 0
                for nn in range(c*GMM_MAX_COMPONT, c*GMM_MAX_COMPONT+GMM_MAX_COMPONT):
                    if m_weight[nn][i][j] != 0:
                        m_weight[nn][i][j] *= weight_scale
                        weight_sum += m_weight[nn][i][j]
                        if abs(img[i][j] - m_mean[nn][i][j]) < 2 * m_sigma[nn][i][j]:
                            cnt += 1
                            break
                        if weight_sum > T:
                            if abs(img[i][j] - m_mean[nn][i][j]) < 2 * m_sigma[nn][i][j]:
                                cnt += 1
                            break
                    else:
                        break


            if cnt == channel:
                m_mask[i][j] = 0

    # cv2.imshow('img_gray', img)
    m_mask = cv2.medianBlur(m_mask, 7)
    # kernel_e = np.ones((3, 3), np.uint8)
    # m_mask = cv2.erode(m_mask, kernel_e)
    kernel_d = np.ones((5, 5), np.uint8)
    m_mask = cv2.dilate(m_mask, kernel_d)
    # element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 形态学去噪
    # m_mask = cv2.morphologyEx(m_mask, cv2.MORPH_OPEN, element)  # 开运算去噪
    return m_mask
    # cv2.imshow('img_mask', m_mask)
    # cv2.waitKey(1)

def test_img(imgs):
    row, col, channel = imgs.shape
    B, G, R = cv2.split(imgs)
    m_mask = np.zeros((row, col), dtype=np.uint8)
    m_mask[:] = 255
    for i in range(row):
        for j in range(col):
            cnt = 0
            for c, img in enumerate((B, G, R)):
                weight_sum = 0
                for nn in range(c * GMM_MAX_COMPONT, c * GMM_MAX_COMPONT + GMM_MAX_COMPONT):
                    if m_weight[nn][i][j] != 0:
                        weight_sum += m_weight[nn][i][j]
                        if abs(img[i][j] - m_mean[nn][i][j]) < 2 * m_sigma[nn][i][j]:
                            cnt += 1
                            break
                        if weight_sum > T:
                            if abs(img[i][j] - m_mean[nn][i][j]) < 2 * m_sigma[nn][i][j]:
                                cnt += 1
                            break
                    else:
                        break

            if cnt == channel:
                m_mask[i][j] = 0

    m_mask = cv2.medianBlur(m_mask, 7)
    kernel_d = np.ones((5, 5), np.uint8)
    m_mask = cv2.dilate(m_mask, kernel_d)

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
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if i == 0:
            init(img)

        if i <= 200:
            t1 = time.time()
            m_mask = train_gmm(img)
            # cv2.imwrite("./rgb_morphologyEx3_MORPH_ELLIPSE/{}.jpg".format(i), m_mask)
            t2 = time.time()
            print(t2-t1)
        if i == 286:
            j = 0
            for temp_file in file_list:
                temp_img = cv2.imread(temp_file)
                m_mask = test_img(temp_img)
                cv2.imwrite("./rgb_mb7_dila5_train200/{:0>5d}.jpg".format(j), m_mask)
                j += 1

