#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

'''
爬取中国语言资源保护工程采录展示平台数据

@see https://zhongguoyuyan.cn
'''

__author__ = '黄艺华 <lernanto@foxmail.com>'


import sys
import logging
import os
import requests
import uuid
import ddddocr
import json
import pandas
import time


site = 'https://zhongguoyuyan.cn'
verify_code_url = '{}/svc/common/create/verifyCode'.format(site)
login_url = '{}/svc/user/login'.format(site)
logout_url = '{}/svc/user/logout'.format(site)
logged_url = '{}/svc/common/validate/logged'.format(site)
standard_url = '{}/svc/api/media/standard'.format(site)
survey_url = '{}/svc/mongo/query/latestSurveyMongo'.format(site)
data_url = '{}/svc/mongo/resource/normal'.format(site)

# 请求一次资源之后的延迟时间
delay = 2


def get_verify_code(session):
    '''请求验证码'''

    logging.info('get verify code from {}'.format(verify_code_url))
    rsp = session.get(verify_code_url)
    logging.debug('cookies = {}'.format(session.cookies))

    return rsp.content

def get_token(image, ocr):
    '''从图片识别验证码'''

    # 使用 OCR 识别图片中的验证码
    token = ocr.classification(image)
    logging.info('get token from image, token = {}'.format(token))
    return token

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
    else:
        # 登录失败
        logging.error('login fialed, code = {}, {}'.format(
            code,
            ret.get('description')
        ))

    return code

def logout(session):
    '''退出登录'''

    logging.info('logout from {}'.format(logout_url))
    session.get(logout_url)

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

def get_standard(session):
    '''获取调查标准'''

    logging.info('get survey standard from {}'.format(standard_url))
    rsp = session.get(standard_url)
    ret = rsp.json()
    logging.debug(ret)

    return ret

def get_survey(session):
    '''获取全部调查点'''

    logging.info('get survey from {}'.format(survey_url))
    rsp = session.post(survey_url)
    survey = rsp.json()
    logging.debug(survey)

    status = survey.get('status')
    if status == 'success':
        logging.info('get survey sucessful')
        return survey
    else:
        logging.error('get survey failed, status = {}'.format(status))

def get_data(session, id):
    '''获取方言点数据'''

    referer = '{}/area_details.html?id={}'.format(site, id)

    logging.info('get data from {}, ID = {}'.format(data_url, id))
    rsp = session.post(data_url, headers={'referer': referer}, data={'id': id})
    ret = rsp.json()
    logging.debug(ret)

    code = ret.get('code')
    if code == 200:
        # 获取数据成功
        logging.debug('get data ID = {} successful, code = {}'.format(id, code))
        return ret.get('data')
    else:
        logging.error('get data ID = {} failed, code = {}, {}'.format(
            id,
            code,
            ret.get('description')
        ))

def try_login(email, password, retry=3):
    '''尝试登录，如果验证码错误，重试多次'''

    session = requests.Session()
    # 网站要求客户端生成一个 UUID 唯一标识当前会话
    session.cookies.set('uniqueVisitorId', str(uuid.uuid4()))

    # 用于识别验证码
    ocr = ddddocr.DdddOcr()

    for i in range(retry):
        # 尝试登录
        image = get_verify_code(session)
        token = get_token(image, ocr)
        code = login(session, email, password, token)

        # 记录一下验证码图片和识别结果
        dir = 'verify_code'
        os.makedirs(dir, exist_ok=True)
        fname = os.path.join(
            dir,
            '.'.join(('_'.join((
                'zhongguoyuyan',
                uuid.uuid4().hex,
                token,
                '0' if code == 702 else '1'
            )), 'jpg'))
        )
        logging.info('save verify code to {}'.format(fname))
        with open(fname, 'wb') as f:
            f.write(image)

        if code != 702:
            # 不是验证码错误，无论登录成功失败都跳出
            break

    if code == 200:
        logging.info('login sucessfule after {} try'.format(i + 1))
        return session
    else:
        logging.error('login failed after {} try, give up'.format(i + 1))

def parse_standard(standard):
    '''解析调查标准 JSON 数据，转换成表格'''

    items = {}
    for key, val in standard.items():
        data = pandas.json_normalize(val)
        items[key] = data

    return items

def parse_survey(survey):
    '''解析调查点 JSON 数据，转换成表格'''

    objects = {}
    for name in ('dialect', 'minority'):
        data = pandas.json_normalize(
            survey[name + 'Obj'],
            'cityList',
            'provinceCode'
        )
        data.index = data['_id']
        objects[name] = data

    return objects

def parse_data(data):
    '''解析方言 JSON 数据，转换成表格'''

    resources = []
    for res in data.get('resourceList', []):
        items = res.get('items')
        if items:
            info = {
                'sounder': res['sounder'],
                'type': res['type'],
                'oid': res['items'][0]['oid']
            }

            record = pandas.json_normalize(items, 'records', ['iid', 'name'])
            record.insert(0, 'iid', record.pop('iid'))
            resources.append((info, record))

    return data['mapLocation'], resources

def crawl_standard(session, prefix='.', update=False, retry=3):
    '''爬取调查标准并保存到文件'''

    # 如果调查标准文件已存在，且指定不更新，则不再重新爬取
    standard_file = os.path.join(prefix, 'standard.json')
    if os.path.exists(standard_file) and not update:
        logging.info(
            'survey standard file {} already exists. do nothing'.format(
                standard_file
            )
        )
        return

    # 调查标准文件不存在，或指定更新文件，从网站获取最新标准
    for i in range(retry):
        standard = get_standard(session)
        time.sleep(delay)
        if standard:
            break

    if standard:
        # 保存调查标准到文件
        logging.info('save survey standard to {}'.format(standard_file))
        with open(standard_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(standard, f, ensure_ascii=False, indent=4)

        for name, item in parse_standard(standard).items():
            fname = os.path.join(prefix, '.'.join((name, '.csv')))
            logging.info('save {} to {}'.format(name, fname))
            item.to_csv(fname, encoding='utf-8', index=False)

def crawl_survey(session, prefix='.', update=False, retry=3):
    '''爬取调查点列表并保存到文件'''

    # 如果调查点列表文件已存在，且指定不更新，则使用现有文件中的数据
    survey_file = os.path.join(prefix, 'survey.json')
    if os.path.exists(survey_file) and not update:
        logging.info(
            'survey data file {} exits. load without crawling'.format(survey_file)
        )
        with open(survey_file, encoding='utf-8') as f:
            survey = json.load(f)

    else:
        # 调查点列表文件不存在，或指定更新文件，从网站获取最新调查点列表
        for i in range(retry):
            survey = get_survey(session)
            time.sleep(delay)
            if survey:
                break

        if survey is None:
            # 超过最大重试次数
            logging.error('cannot get survey after {} try, give up'.format(i + 1))
            return

        logging.info('save survey data to {}'.format(survey_file))
        with open(survey_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(survey, f, ensure_ascii=False, indent=4)

    return parse_survey(survey)

def crawl_data(session, id, prefix='.', update=False, retry=3):
    '''爬取一个调查点的数据并保存到文件'''

    data_file = os.path.join(prefix, '{}.json'.format(id))
    if os.path.exists(data_file) and not update:
        # 数据文件已存在，且指定不更新，不再重复爬取
        logging.info(
            'data file {} already exists. do nothing'.format(data_file)
        )
        return

    for i in range(retry):
        # 请求调查点数据
        data = get_data(session, id)
        time.sleep(delay)
        if data:
            logging.info('save data to {}'.format(data_file))
            with open(data_file, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            return True

    # 超过最大重试次数
    logging.error('cannot get data ID = {} after {} try, give up'.format(
        id,
        i + 1
    ))
    return False

def crawl(
    email,
    password,
    prefix='.',
    update_survey=False,
    max_success=300,
    max_fail=3
):
    '''爬取调查点数据'''

    success_count = 0
    fail_count = 0

    logging.info('try creating directory {}'.format(prefix))
    os.makedirs(prefix, exist_ok=True)

    # 登录网站
    session = try_login(email, password)
    if session:
        # 爬取调查标准
        crawl_standard(session, prefix)

        # 爬取调查点列表
        ret = crawl_survey(session, prefix)

        if ret:
            dialect, minority = ret
            minority_dir = os.path.join(prefix, 'minority')
            logging.info('try creating minority directory {}'.format(minority_dir))
            os.makedirs(minority_dir, exist_ok=True)

            stop = False
            for obj, name in ((dialect, 'dialect'), (minority, 'minority')):
                dir = os.path.join(prefix, name)
                logging.info('try creating directory {}'.format(dir))
                os.makedirs(dir, exist_ok=True)

                for id, row in obj.iterrows():
                    # 爬取一个调查点的数据
                    logging.info('get {} ID = {}, city = {}'.format(
                        name,
                        id,
                        row['city'])
                    )
                    ret = crawl_data(session, id, dir)

                    if ret is not None:
                        if ret:
                            success_count += 1
                            if success_count % 100 == 0:
                                logging.info('crawled {} data'.format(success_count))

                            if success_count >= max_success:
                                # 达到设置的最大爬取数量
                                logging.info(
                                    'reached maximum crawl number = {}, have a rest'.format(
                                        max_success
                                    )
                                )
                                stop = True
                                break

                        else:
                            fail_count += 1
                            if fail_count >= max_fail:
                                # 多次爬取失败，不再尝试
                                logging.error(
                                    'reached maximum failure number = {}, abort'.format(
                                        max_fail
                                    )
                                )
                                stop = True
                                break

                if stop:
                    break

            if not stop:
                logging.info(
                    'all data crawled, nothing else todo, success_count = {}'.format(
                        success_count
                    )
                )

        else:
            # 获取调查点列表失败
            logging.error('cannot get survey. exit')

        # 退出登录
        logout(session)

    logging.info('totally crawl {} data'.format(success_count))

def parse(indir='.', outdir='.', update=False):
    '''把 JSON 格式的原始数据转换成 CSV 格式'''

    logging.info(
        'parse data from JSON to CSV, input directory = {}, output directory = {}'.format(
            indir,
            outdir
        )
    )

    # 如果输出目录不存在，先创建
    logging.info('try creating directory {}'.format(outdir))
    os.makedirs(outdir, exist_ok=True)

    # 解析调查标准文件
    standard_file = os.path.join(indir, 'standard.json')
    logging.info('parse standard file {}'.format(standard_file))
    with open(standard_file, encoding='utf-8') as f:
        standard = json.load(f)

    for name, item in parse_standard(standard).items():
        fname = os.path.join(outdir, name + '.csv')
        logging.info('save {} to {}'.format(name, fname))
        item.to_csv(fname, index=False, encoding='utf-8', line_terminator='\n')

    # 解析调查点列表文件
    survey_file = os.path.join(indir, 'survey.json')
    logging.info('parse survey data file {}'.format(survey_file))
    with open(survey_file, encoding='utf-8') as f:
        survey = json.load(f)

    lists = parse_survey(survey)
    for name, lst in lists.items():
        # 保存调查点列表
        fname = os.path.join(outdir, name + '.csv')
        logging.info('save {} list to {}'.format(name, fname))
        lst.to_csv(fname, encoding='utf-8', line_terminator='\n')

        prefix = os.path.join(outdir, name)
        logging.info('try creating directory {}'.format(prefix))
        os.makedirs(prefix, exist_ok=True)

        # 从文件读取解析所有调查点数据
        ids = []
        location = []

        for id, row in lst.iterrows():
            fname = os.path.join(indir, name, id + '.json')
            if os.path.exists(fname):
                logging.debug('parse data file {}'.format(fname))
                with open(fname, encoding='utf-8') as f:
                    data = json.load(f)

                loc, resources = parse_data(data)
                ids.append(id)
                location.append(dict(**loc['point'], **loc['location']))

                # 把单字、词汇、语法等数据分别保存成 CSV 文件
                for info, res in resources:
                    fname = os.path.join(prefix, info['oid'] + '.csv')
                    if os.path.exists(fname) and not update:
                        logging.info(
                            'resource file {} already exists. do nothing'.format(
                                fname
                            )
                        )
                    else:
                        logging.info('save resource to {}'.format(fname))
                        res.to_csv(
                            fname,
                            index=False,
                            encoding='utf-8',
                            line_terminator='\n'
                        )

        location = pandas.DataFrame(location, index=ids)
        logging.debug(location)

        fname = os.path.join(prefix, 'location.csv')
        logging.info('save location data to {}'.format(fname))
        location.to_csv(fname, encoding='utf-8', line_terminator='\n')


def main():
    logging.getLogger().setLevel(logging.INFO)

    email, password, prefix, update_survey, max_success = sys.argv[1:6]
    crawl(email, password, prefix)


if __name__ == '__main__':
    main()