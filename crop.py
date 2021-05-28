# -*- encoding=utf8 -*-
import cv2
import os
import numpy as np
import capture

# ultimate_positions = [[(692, 164), (788, 164), (884,164), (692, 266)]]
# print(ultimate_positions[0][0])

ultimate_positions = [[], []]
for i in range(2):
    for j in range(6):
        if i==1 and j == 0:
            ultimate_positions[i].append((696 + j * 96, 166 + i * 100))
        else:
            ultimate_positions[i].append((692 + j * 96, 166 + i * 100))
print(ultimate_positions)
skill_positions = [[],[],[],[],[],[]]
for i in range(6):
    for j in range(6):
        skill_positions[i].append((730 + j * 78, 343 + i * 100))


ultimate_list = []
skill_image_list = []

target = cv2.imread(r'target2.jpg')
for i in range(2):
    for j in range(6):
        ultimate_list.append(target[ultimate_positions[i][j][1]:ultimate_positions[i][j][1] + 55,
                             ultimate_positions[i][j][0]:ultimate_positions[i][j][0] + 55])
for i in range(6):
    for j in range(6):
        skill_image_list.append(target[skill_positions[i][j][1]:skill_positions[i][j][1] + 40 + j*2,
                                skill_positions[i][j][0]:skill_positions[i][j][0] + 45])

# ultimates = np.hstack(skill_image_list)
# cv2.imshow('123', ultimates)
# print(skill_positions[0][0])
# cv2.imshow('123', skill_image_list[2])
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

def ahash(img):
    #缩放为8*8
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    #转换为灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #s为像素和初值为0，hash_str为hash值初值为''
    s=0
    hash_str=''
    #遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s=s+gray[i,j]
    #求平均灰度
    avg=s/64
    #灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if  gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

def phash(image):
    # 感知哈希算法
    # 缩放32*32
    img = cv2.resize(image, (32, 32))  # , interpolation=cv2.INTER_CUBIC

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


def dhash(img):
    # 差值哈希算法
    # 缩放8*8
    img = cv2.resize(img, (9, 8))
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


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

test_phash = phash(skill_image_list[0])
test_max = 40
test_phash_real = ''
for skill in os.listdir(r'.\skill_image'):
    test_compare = campHash(test_phash, phash(cv2.imread('.\\skill_image\\'+skill)))
    if test_compare < test_max:
        test_max = test_compare
        test_phash_real = skill
print(test_phash_real)
    


# skill_list = capture.get_skill_list()
# ultimate_positions_list = sum(ultimate_positions, [])
# for i in range(len(ultimate_list)):
#     skill_ahash = ahash(ultimate_list[i])
#     skill_phash = phash(ultimate_list[i])
#     skill_dhash = dhash(ultimate_list[i])
#
#     max_ahash = 40
#     max_phash = 40
#     max_dhash = 40
#     ahash_real = ''
#     phash_real = ''
#     dhash_real = ''
#
#
#     for skill in os.listdir(r'.\ultimate_image'):
#         skill_image = cv2.imread('.\\ultimate_image\\' + skill)
#         # compare = classify_hist_with_split(skill_image, ultimate_list[i])
#         ahash_compare = campHash(skill_ahash, ahash(skill_image))
#         phash_compare = campHash(skill_phash, phash(skill_image))
#         dhash_compare = campHash(skill_dhash, dhash(skill_image))
#         # # if skill == 'disruptor_static_storm.jpg':
#         # print('c=' + str(max_ahash))
#         # print('c=' + str(max_phash))
#         # print('c=' + str(max_dhash))
#         if ahash_compare < max_ahash:
#             max_ahash = ahash_compare
#             ahash_real = skill
#         if phash_compare < max_phash:
#             max_phash = phash_compare
#             phash_real = skill
#         if dhash_compare < max_dhash:
#             max_dhash = dhash_compare
#             dhash_real = skill
#
#     real = ''
#
#     if ahash_real == phash_real:
#         real = ahash_real
#     elif ahash_real == dhash_real:
#         real = ahash_real
#     elif phash_real == dhash_real:
#         real = phash_real
#     elif min(max_ahash,max_dhash-2,max_phash) == max_ahash:
#         real = ahash_real
#     elif min(max_ahash,max_dhash-2,max_phash) == max_phash:
#         real = phash_real
#     elif min(max_ahash,max_dhash-2,max_phash) == max_dhash-2:
#         real = dhash_real
#
#
#
#
#
#     # print(ahash_real)
#     # print('c=' + str(max_ahash))
#     #
#     # print(phash_real)
#     # print('c=' + str(max_phash))
#     #
#     # print(dhash_real)
#     # print('c=' + str(max_dhash))
#     # if min(max_ahash,max_dhash-2,max_phash) == max_ahash:
#     #     print(ahash_real)
#     # elif min(max_ahash,max_dhash-2,max_phash) == max_phash:
#     #     print(phash_real)
#     # elif min(max_ahash,max_dhash-2,max_phash) == max_dhash-2:
#     #     print(dhash_real)
#     cv2.putText(target, skill_list[real.split('.')[0]]['winRate']+'%', (ultimate_positions_list[i][0], ultimate_positions_list[i][1]-15),
#                 cv2.FONT_HERSHEY_COMPLEX, 0.4, (84, 255, 159), 1, cv2.LINE_AA)
#     cv2.putText(target, '(' + skill_list[real.split('.')[0]][
#         'order'] + ')', (ultimate_positions_list[i][0], ultimate_positions_list[i][1]),
#                 cv2.FONT_HERSHEY_COMPLEX, 0.4, (84, 255, 159), 1, cv2.LINE_AA)
#     target = capture.cv2ImgAddText(target, skill_list[real.split('.')[0]]['name'], ultimate_positions_list[i][0], ultimate_positions_list[i][1]+55,(84, 255, 159),15)
# cv2.imshow('',target)
# cv2.waitKey()
# cv2.destroyAllWindows()



