# -*- encoding=utf8 -*-
import os
import json
import time
import logging
import requests

def get_skill_rank():
    if not os.path.exists('skill_rank.json') \
            or time.time() - os.path.getmtime('skill_rank.json') > 604800: # 文件修改时间大于一周
        session_request = requests.session()
        login_url = 'https://dota2omg.com/login'
        login_result = session_request.post(
            login_url,
            data={
                'username': 'guest',
                'password': 'guest',
                'rememberMe': 'false',
            },
        )
        if json.loads(login_result.content.decode('utf-8'))['code'] != 0:
            logging.error('访客登录失败，请检查网页是否可以正常访问')
        else:
            logging.info('访客登陆成功')
            rate_url = 'https://dota2omg.com/winrate/getAbilitiesWinrate'
            rate_result = session_request.get(rate_url)
            # print(rate_result.text)
            with open('skill_rank.json', 'w') as fp:
                j = json.dumps(json.loads(rate_result.content.decode('utf-8'))['rows'],indent=4)
                fp.write(j)
    with open('skill_rank.json', 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        return json_data



if  __name__ == '__main__':
    print(get_skill_rank())
