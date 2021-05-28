# -*- encoding=utf8 -*-
import cv2
import os
import numpy as np

# ultimate_positions = [[(692, 164), (788, 164), (884,164), (692, 266)]]
# print(ultimate_positions[0][0])

ultimate_positions = [[],[]]
for i in range(2):
    for j in range(6):
            ultimate_positions[i].append((692+j*96, 164+i*102))
print(ultimate_positions)


ultimate_list = []

target = cv2.imread(r'D:\PyCharm\PycharmProjects\omg_analysis\target2.jpg')
i = 2
for i in range(2):
    for j in range(6):
        ultimate_list.append(target[ultimate_positions[i][j][1]:ultimate_positions[i][j][1] + 60,
            ultimate_positions[i][j][0]:ultimate_positions[i][j][0] + 60])
# ultimates = np.hstack(ultimate_list)
# cv2.imshow('123', ultimates)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def calculate(image1, image2):
    # 灰度直方图算法
    # 计算单通道的直方图的相似值
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

def classify_hist_with_split(image1, image2, size=(256, 256)):
    # RGB每个通道的直方图相似度
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data

def phash(img):
    # 感知哈希算法
    # 缩放32*32
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(gray))
    # opencv实现的掩码操作
    dct_roi = dct[0:8, 0:8]

    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash



def campHash(hash1, hash2):
    n = 0
    # hash长度不同返回-1,此时不能比较
    if len(hash1) != len(hash2):
        return -1
    # 如果hash长度相同遍历长度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


for i in range(len(ultimate_list)):
    skill_hash = phash(ultimate_list[i])

    maxHash = 0
    real = ''
    for skill in os.listdir(r'.\skill_image'):
        skill_image = cv2.imread('.\skill_image\\'+skill)
        compare = classify_hist_with_split(skill_image, ultimate_list[i])
        #compare = campHash(skill_hash, a)
        if compare > maxHash:
            maxHash = compare
            real = skill


    print(real)
    print(maxHash)




