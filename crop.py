# -*- encoding=utf8 -*-

import pynput
import time
import cv2
import os
import numpy as np
import capture
import skill_rank
import pyperclip


from multiprocessing import Pool
from itertools import chain

ORANGE = (231, 101, 26)
PURPLE = (218, 112, 214)
BLUE = (0, 153, 204)
GREEN = (153, 204, 0)
GRAY = (192, 192, 192)

# ultimate_positions = [[(692, 164), (788, 164), (884,164), (692, 266)]]
# print(ultimate_positions[0][0])


# for i in range(6):
#     for j in range(6):
#         if j >= 3:
#             distance = 25
#         else:
#             distance = 0
#         skill_positions[i].append((730 + j * 78+ distance, 343 + i * 68))


# target = cv2.imread(r'target3.jpg')



# ultimates = np.hstack(skill_image_list)
# cv2.imshow('123', ultimates)
# print(skill_positions[0][0])
# cv2.imshow('123', skill_image_list[32])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#



def ahash(img):
    # 缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
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

def skill_match(skill_image):
    hash = phash(skill_image)
    compare_min = 25
    match_skill = ''
    for image in os.listdir(r'.\skill_image'):
        compare = campHash(hash, phash(cv2.imread('.\\skill_image\\' + image)))
        if compare < compare_min:
            compare_min = compare
            match_skill = image.split('.')[0]
    return match_skill

def ultimate_match(ultimate_image):
    hash = phash(ultimate_image)
    compare_min = 25
    match_skill = ''
    for image in os.listdir(r'.\ultimate_image'):
        compare = campHash(hash, phash(cv2.imread('.\\ultimate_image\\' + image)))
        if compare < compare_min:
            compare_min = compare
            match_skill = image.split('.')[0]

    return match_skill

def print_screen():
    print('截图成功')
    raise Exception


if __name__ == '__main__':
    ultimate_positions = [[], []]
    for i in range(2):
        for j in range(6):
            if i == 1 and j == 0:
                ultimate_positions[i].append((696 + j * 96, 166 + i * 100))
            else:
                ultimate_positions[i].append((692 + j * 96, 166 + i * 100))
    skill_positions = [[(732, 345), (808, 345), (884, 345), (987, 345), (1064, 345), (1140, 345)],
                       [(725, 410), (804, 410), (883, 410), (988, 410), (1066, 410), (1146, 410)],
                       [(718, 479), (799, 479), (882, 479), (989, 479), (1070, 479), (1152, 479)],
                       [(708, 594), (792, 594), (881, 594), (993, 594), (1079, 594), (1163, 594)],
                       [(702, 674), (791, 674), (879, 674), (993, 674), (1081, 674), (1169, 674)],
                       [(694, 759), (785, 759), (874, 759), (996, 759), (1086, 759), (1177, 759)]]

    skill_list = skill_rank.get_skill_rank()
    while 1:
        try:
            print('请在选择技能界面按下截图键PrintScreen：')
            with pynput.keyboard.GlobalHotKeys({
                '<print_screen>':print_screen,
            }) as h:
                h.join()
        except Exception:
            time.sleep(2)
            ultimate_list = []
            skill_image_list = []
            skill_pool = {}
            target = capture.get_clipboard_picture()
            pyperclip.copy('')
            for i in range(2):
                for j in range(6):
                    ultimate_list.append(target[ultimate_positions[i][j][1]:ultimate_positions[i][j][1] + 55,
                                         ultimate_positions[i][j][0]:ultimate_positions[i][j][0] + 55])
            for i in range(6):
                for j in range(6):
                    skill_image_list.append(target[skill_positions[i][j][1]:skill_positions[i][j][1] + 42 + i * 3,
                                            skill_positions[i][j][0]:skill_positions[i][j][0] + 48 + j * 1])

            t0 = time.time()
            # skill_phashp_list = []
            # for m in range(6):
            #     for i in skill_positions[m]:
            #         skill_phashp_list.append(i)

            pool = Pool(10)
            normal_pool = pool.map(skill_match, skill_image_list)
            pool.close()
            pool.join()
            for skill in normal_pool:
                skill_pool[skill] = skill_list[skill]

            # for i in range(len(skill_image_list)):
            #     test_ahash = ahash(skill_image_list[i])
            #     test_phash = phash(skill_image_list[i])
            #     test_dhash = dhash(skill_image_list[i])
            #     test_a_max = 40
            #     test_ahash_real = ''
            #     test_p_max = 40
            #     test_phash_real = ''
            #     test_d_max = 40
            #     test_dhash_real = ''
            #
            #     for skill in os.listdir(r'.\skill_image'):
            #         # test_compare_a = campHash(test_ahash, ahash(cv2.imread('.\\skill_image\\' + skill)))
            #         test_compare_p = campHash(test_phash, phash(cv2.imread('.\\skill_image\\' + skill)))
            #         # test_compare_d = campHash(test_dhash, dhash(cv2.imread('.\\skill_image\\' + skill)))
            #         # if test_compare_a < test_a_max:
            #         #     test_a_max = test_compare_a
            #         #     test_ahash_real = skill
            #         if test_compare_p < test_p_max:
            #             test_p_max = test_compare_p
            #             test_phash_real = skill
            #         # if test_compare_d < test_d_max:
            #         #     test_d_max = test_compare_d
            #         #     test_dhash_real = skill
            #     real = test_phash_real
            #     # if test_ahash_real == test_phash_real:
            #     #     real = test_ahash_real
            #     # elif test_ahash_real == test_dhash_real:
            #     #     real = test_ahash_real
            #     # elif test_phash_real == test_dhash_real:
            #     #     real = test_phash_real
            #     # elif min(test_a_max, test_d_max-2, test_p_max) == test_a_max:
            #     #     real = test_ahash_real
            #     # elif min(test_a_max, test_d_max-2, test_p_max) == test_p_max:
            #     #     real = test_phash_real
            #     # elif min(test_a_max, test_d_max-2, test_p_max) == test_d_max - 2:
            #     #     real = test_dhash_real
            #     skill_pool[real.split('.')[0]] = skill_list[real.split('.')[0]]
            #     # print(test_phash_real)
            #     # print(test_max)
            #     color = (0, 0, 0)
            #     order = int(skill_list[real.split('.')[0]]['order'])
            #     if order <= 48:
            #         color = ORANGE
            #     elif order > 48 and order < 145:
            #         color = PURPLE
            #     elif order >=145 and order <339:
            #         color = BLUE
            #     elif order >=339 and order < 453:
            #         color = GREEN
            #     elif order >=453:
            #         color = GRAY
            #     # cv2.putText(target,
            #     #             skill_list[real.split('.')[0]]['winRate'] + '%(' + skill_list[real.split('.')[0]][
            #     #                 'order'] + ')',
            #     #             (skill_phashp_list[i][0], skill_phashp_list[i][1]),
            #     #             cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            #     # cv2.putText(target, , (skill_phashp_list[i][0], skill_phashp_list[i][1]),
            #     #             cv2.FONT_HERSHEY_COMPLEX, 0.4, (84, 255, 159), 1, cv2.LINE_AA)
            #
            #     target = capture.cv2ImgAddText(target, skill_list[real.split('.')[0]]['name']+'('+str(order)+')', skill_phashp_list[i][0]-8,
            #                                    skill_phashp_list[i][1] + 43 + i / 6 * 3, color, 12)

            # print('耗时' + str(time.time() - t0))
            # cv2.imshow('123', skill_image_list[5])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            ultimate_positions_list = sum(ultimate_positions, [])

            t1 = time.time()
            for i in range(len(ultimate_list)):
                skill_ahash = ahash(ultimate_list[i])
                skill_phash = phash(ultimate_list[i])
                skill_dhash = dhash(ultimate_list[i])

                max_ahash = 40
                max_phash = 40
                max_dhash = 40
                ahash_real = ''
                phash_real = ''
                dhash_real = ''

                for skill in os.listdir(r'.\ultimate_image'):
                    skill_image = cv2.imread('.\\ultimate_image\\' + skill)
                    # compare = classify_hist_with_split(skill_image, ultimate_list[i])
                    ahash_compare = campHash(skill_ahash, ahash(skill_image))
                    phash_compare = campHash(skill_phash, phash(skill_image))
                    dhash_compare = campHash(skill_dhash, dhash(skill_image))
                    # # if skill == 'disruptor_static_storm.jpg':
                    # print('c=' + str(max_ahash))
                    # print('c=' + str(max_phash))
                    # print('c=' + str(max_dhash))
                    if ahash_compare < max_ahash:
                        max_ahash = ahash_compare
                        ahash_real = skill
                    if phash_compare < max_phash:
                        max_phash = phash_compare
                        phash_real = skill
                    if dhash_compare < max_dhash:
                        max_dhash = dhash_compare
                        dhash_real = skill

                real = ''

                if ahash_real == phash_real:
                    real = ahash_real
                elif ahash_real == dhash_real:
                    real = ahash_real
                elif phash_real == dhash_real:
                    real = phash_real
                elif min(max_ahash, max_dhash - 2, max_phash) == max_ahash:
                    real = ahash_real
                elif min(max_ahash, max_dhash - 2, max_phash) == max_phash:
                    real = phash_real
                elif min(max_ahash, max_dhash - 2, max_phash) == max_dhash - 2:
                    real = dhash_real

                skill_pool[real.split('.')[0]] = skill_list[real.split('.')[0]]
                # print(ahash_real)
                # print('c=' + str(max_ahash))
                #
                # print(phash_real)
                # print('c=' + str(max_phash))
                #
                # print(dhash_real)
                # print('c=' + str(max_dhash))
                # if min(max_ahash,max_dhash-2,max_phash) == max_ahash:
                #     print(ahash_real)
                # elif min(max_ahash,max_dhash-2,max_phash) == max_phash:
                #     print(phash_real)
                # elif min(max_ahash,max_dhash-2,max_phash) == max_dhash-2:
                #     print(dhash_real)
                color = (0, 0, 0)
                order = int(skill_list[real.split('.')[0]]['order'])
                if order <= 48:
                    color = ORANGE
                elif order > 48 and order < 145:
                    color = PURPLE
                elif order >= 145 and order < 339:
                    color = BLUE
                elif order >= 339 and order < 453:
                    color = GREEN
                elif order >= 453:
                    color = GRAY
                if int(skill_list[real.split('.')[0]]['order']) < 10:
                    color = ORANGE
                # cv2.putText(target, skill_list[real.split('.')[0]]['winRate'] + '%(' + skill_list[real.split('.')[0]][
                #     'order'] + ')', (ultimate_positions_list[i][0], ultimate_positions_list[i][1] - 5),
                #             cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                # cv2.putText(target, , (ultimate_positions_list[i][0], ultimate_positions_list[i][1]),
                #             cv2.FONT_HERSHEY_COMPLEX, 0.4, (84, 255, 159), 1, cv2.LINE_AA)
                target = capture.cv2ImgAddText(target, skill_list[real.split('.')[0]]['name'] + '(' + str(order) + ')',
                                               ultimate_positions_list[i][0] - 8,
                                               ultimate_positions_list[i][1] + 55, color, 12)

            # pool2 = Pool(10)
            # ultimate_pool = pool2.map(ultimate_match, ultimate_list)
            # pool2.close()
            # pool2.join()
            # for skill in ultimate_pool:
            #     skill_pool[skill] = skill_list[skill]


            print('耗时' + str(time.time() - t0))
            x = 0
            skill_pool = sorted(skill_pool.items(), key=lambda x: x[1]['mixOrder'])
            print('本局推荐先选的技能为:')
            text = ''
            for i in skill_pool:
                x += 1
                text += 'No.' + str(x) + ' ' + dict(i[1])['name'] +'\n'
                if x == 12:
                    break
            print(text)
            pyperclip.copy(text)
            # import pygame
            # pygame.mixer.init()
            # pygame.mixer.music.load('complete.wav')
            # pygame.mixer.music.set_volume(1)
            # pygame.mixer.music.play()
            import playsound
            playsound.playsound('complete.wav')
            #pyperclip.paste()


            # def get_top_skill(pool):
            #     top_skills = []
            #     min_mix_order = 20
            #     for i in pool:
            #         if len(top_skills) > 10:
            #             top_skills
            #
            #
            #
            #
            #     for i in top_skills:

            #
            # cv2.imshow('', target)
            # cv2.waitKey()
            # cv2.destroyAllWindows()