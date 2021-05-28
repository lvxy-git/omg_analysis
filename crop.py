# -*- encoding=utf8 -*-
import cv2
import os
import numpy

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
# ultimates = numpy.hstack(ultimate_list)
# cv2.imshow('123', ultimates)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def ahash(image):
    # 将图片缩放为8*8的
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 将图片转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # s为像素和初始灰度值，hash_str为哈希值初始值
    s = 0
    # 遍历像素累加和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 计算像素平均值
    avg = s / 64
    # 灰度大于平均值为1相反为0，得到图片的平均哈希值，此时得到的hash值为64位的01字符串
    ahash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                ahash_str = ahash_str + '1'
            else:
                ahash_str = ahash_str + '0'
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(ahash_str[i: i + 4], 2))
    # print("ahash值：",result)
    return result

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
    skill_hash = ahash(ultimate_list[i])

    maxHash = 20
    real = ''
    for skill in os.listdir(r'.\skill_image'):
        skill_image = cv2.imread('.\skill_image\\'+skill)
        a = ahash(skill_image)
        compare = campHash(skill_hash, a)
        # print(compare)
        if compare < maxHash:
            maxHash = compare
            real = skill
    if maxHash<=8:
        print(real)
        print(maxHash)




