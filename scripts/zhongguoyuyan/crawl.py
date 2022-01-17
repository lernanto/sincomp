#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

'''
爬取中国语言资源保护工程采录展示平台数据

@see https://zhongguoyuyan.cn
'''

__author__ = '黄艺华 <lernanto@foxmail.com>'


import sys
import logging
import requests
import uuid
import json


site = 'https://zhongguoyuyan.cn'
verify_code_url = '{}/svc/common/create/verifyCode'.format(site)
login_url = '{}/svc/user/login'.format(site)
logged_url = '{}/svc/common/validate/logged'.format(site)
data_url = '{}/svc/mongo/resource/normal'.format(site)


def get_verify_code(session):
    '''请求验证码'''

    logging.info('get verify code from {}'.format(verify_code_url))
    rsp = session.get(verify_code_url)
    logging.debug('cookies = {}'.format(session.cookies))

    return rsp

def get_token(code):
    '''从图片识别验证码'''

    # TODO
    pass

def login(session, email, password, token):
    '''登录到网站'''

    # 发送登录请求
    logging.info('login from {}, email = {}, pass = {}, token = {}'.format(
        login_url,
        email,
        password,
        token
    ))
    rsp = session.post(
        login_url,
        data={'email': email, 'pass': password, 'token': token}
    )
    ret = rsp.json()
    logging.debug(ret)
    logging.debug('cookies = {}'.format(session.cookies))

    code = ret.get('code')
    if code == 200:
        # 登录成功
        logging.info('login successful, code = {}'.format(code))
        return True
    else:
        # 登录失败
        logging.error('login fialed, code = {}, {}'.format(
            code,
            ret.get('description')
        ))
        return False

def validate_login(session):
    '''验证登录状态'''

    # 验证是否登录成功
    logging.info('validate login from {}'.format(logged_url))
    rsp = session.post(logged_url)
    ret = rsp.json()
    logging.debug(ret)
    logging.debug('cookies = {}'.format(session.cookies))

    if ret.get('loginFlag') == 1:
        # 用户已经登录，输出用户信息
        user = ret.get('user', {})
        logging.info('user logged in, user ID = {}, E-mail = {}, last logged in at {}'.format(
            user.get('user_id'),
            user.get('email'),
            user.get('login_record', [{}])[-1].get('time')
        ))
        return True
    else:
        # 用户未登录
        logging.info('user not logged in')
        return False

def get_data(session, id):
    '''获取方言点数据'''

    referer = '{}/area_details.html?id={}'.format(site, id)

    logging.info('get data from {}, ID = {}'.format(data_url, id))
    rsp = session.post(data_url, headers={'referer': referer}, data={'id': id})
    ret = rsp.json()
    logging.debug(ret)
    logging.debug('cookies = {}'.format(session.cookies))

    code = ret.get('code')
    if code == 200:
        # 获取数据成功
        logging.info('get data successful, code = {}'.format(code))
        return ret.get('data')
    else:
        logging.error('get data failed, code = {}, {}'.format(
            code,
            ret.get('description')
        ))

def main():
    logging.getLogger().setLevel(logging.INFO)

    email, password, id = sys.argv[1:4]

    session = requests.Session()
    session.cookies.set('uniqueVisitorId', str(uuid.uuid4()))

    rsp = get_verify_code(session)
    token = get_token(rsp.content)
    if login(session, email, password, token):
        data = get_data(session, id)
        if data:
            json.dump(data, sys.stdout, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()