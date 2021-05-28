# -*- coding: utf-8 -*-
import json
import xlrd
from xlutils.copy import copy
import requests

id_list = ['144867662', '132431263', '172996673', '176171925', '364680852', '181470250',
           '172460363', '136517497', '130776494', '268964620', '234372027', '214739833']

session_request = requests.session()
login_url = 'http://dota2omg.com/login'
result = session_request.post(
    login_url,
    data={
        'username': 'guest',
        'password': 'guest',
        'rememberMe': 'false',
    },
    headers=dict(referer=login_url)
)

rank_excel = xlrd.open_workbook('rank.xls', formatting_info=True)
sheet = rank_excel.sheet_by_index(0)
new_rank_excel = copy(rank_excel)
new_sheet = new_rank_excel.get_sheet(0)
n = 0
rank_url = 'http://dota2omg.com/account/getAccountInfo/'  #
for id in id_list:
    n += 1
    rank_result = session_request.post(rank_url,
                                       data={
                                           'accountId': id,
                                       },
                                       headers=dict(referer=rank_url))
    account_data = json.loads(rank_result.content.decode('utf-8'))['data']['accountInfo']
    # print(account_data)
    new_sheet.write(n, 0, account_data['personaName'].encode('GBK', 'ignore').decode('GBk'))
    print('username:' + account_data['personaName'].encode('GBK', 'ignore').decode('GBk'))
    new_sheet.write(n, 1, str(account_data['omgMmr']).split('.')[0])
    print('omg_score:' + str(account_data['omgMmr']).split('.')[0])
    new_sheet.write(n, 2, str(account_data['performanceMmr']).split('.')[0])
    print('performance_score:' + str(account_data['performanceMmr']).split('.')[0])
    new_sheet.write(n, 3, str(account_data['kda'])[:-2])
    print('kda:' + str(account_data['kda'])[:-2])

try:
    new_rank_excel.save('./rank.xls')
except PermissionError:
    print('\n该Excel文件当前已被打开，请关闭后再次重试！！！')