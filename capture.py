import numpy
import os
import cv2
import json
import requests
from PIL import Image, ImageDraw, ImageFont
from airtest import aircv


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


def get_skill_list():
    session_request = requests.session()
    login_url = 'https://dota2omg.com/login'
    result = session_request.post(
        login_url,
        data={
            'username': 'guest',
            'password': 'guest',
            'rememberMe': 'false',
        },
        # headers=dict(referer=login_url)
    )
    # print(session_request.cookies)

    # print(result.text)
    rate_url = 'https://dota2omg.com/winrate/getAbilitiesWinrate'  #
    rate_result = session_request.get(rate_url)
    # print(rate_result.content)

    if not os.path.exists(r'.\skill_image'):
        os.mkdir(r'.\skill_image')

    skill_list = {}
    # print(rate_result.text)
    n = 0
    for i in json.loads(rate_result.content.decode('utf-8'))['rows']:
        if True:
            # print(i)
            n += 1
            skill_list[i['nameEn']] = {
                'order': str(n),
                'name': i['nameCn'],
                'winRate': str(i['winrate'] * 100)[:4],
                'pickOrder': i['pickOrder']}

    # print(skill_list)
    return skill_list
    # if not os.path.exists(".\\skill_image\\"+i['nameEn']+'.jpg'):
    #     image_result = session_request.get('http://dota2omg.com'+i['url'])
    #     with open(".\\skill_image\\"+i['nameEn']+'.jpg','wb') as f:
    #         f.write(image_result.content)
    #         f.close()
    # template = cv2.imread(".\\skill_image\\" + i['nameEn'] + '.jpg ')
    # try:
    #     result = aircv.find_template(target, template, threshold=0.8, rgb=True)
    #     if not result:
    #         result = aircv.find_sift(target, template, threshold=0.8, rgb=True)
    # except:
    #     pass
    # if result:
    #     cv2.rectangle(target, result['rectangle'][0], result['rectangle'][2], (0, 0, 225), 2)

    #     cv2.rectangle(target, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (0, 0, 225), 2)
    #     cv2.putText(target, str(round(i['winrate']*100,0))+'%',(min_loc[0],min_loc[1]+theight),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)


# cv2.imshow("Result", target)
# cv2.waitKey()
# cv2.destroyAllWindows()

from PIL import Image, ImageGrab

def get_clipboard_picture():
    im = ImageGrab.grabclipboard()

    if isinstance(im, Image.Image):
        print("Image: size : %s, mode: %s" % (im.size, im.mode))
        im.save("clipboard.jpg")
    elif im:
        for filename in im:
            try:
                print("filename: %s" % filename)
                im = Image.open(filename)
            except IOError:
                pass #ignore this file
            else:
                print("ImageList: size : %s, mode: %s" % (im.size, im.mode))
    else:
        print("clipboard is empty.")

if __name__ == '__main__':
    get_clipboard_picture()
