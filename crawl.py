#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

'''
从汉字古今音资料库爬取方言字音.

该库为广韵每个字分配了 ID，可根据 ID 和大方言查询各方言点读音。
根据输入的 ID 和指定的方言爬取读音。输出为 JSON 格式，每字一行。

@see http://xiaoxue.iis.sinica.edu.tw/ccr
'''

__author__ = 'Edward Wong <lernanto.wong@gmail.com>'


import sys
import logging
import urllib
import requests
import lxml.etree
import json
import time


def main():
    logging.basicConfig(
        format='[%(levelname).1s] %(asctime)s %(message)s',
        level=logging.INFO
    )

    # 参数指定爬取的方言，如 yueyu
    lang = sys.argv[1]
    url = r'http://xiaoxue.iis.sinica.edu.tw/{}/PageResult/PageResult'.format(lang)

    # 从标准输入读入爬取的字 ID，范围1 - 25528
    count = 0
    for line in sys.stdin:
        try:
            cid = int(line)
            rsp = requests.post(url, data={'ZiOrder': cid})
            selector = lxml.etree.HTML(rsp.text)
            assert int(selector.xpath(r'string(//span[@id="StartOrder"])')) == cid

            # 获取字形
            img = selector.xpath(r'//td[@class="ZiList"]/../td[2]/img/@src')[0]
            logging.debug(img)
            char = urllib.parse.parse_qs(urllib.parse.urlparse(img).query)['text'][0]
            data = {'id': cid, 'char': char}

            # 中古汉语声韵调
            mc = selector.xpath(r'//table[2]/tr')
            logging.debug(lxml.etree.tostring(mc[2]))
            data['middle_chinese'] = dict(zip(
                [c.xpath(r'string()').strip() for c in mc[1].xpath(r'td')[1:]],
                [c.xpath(r'string()').strip() for c in mc[2].xpath(r'td')[1:]]
            ))

            # 现代方言读音表，每行一个方言点
            rows = selector.xpath(r'//table[@id="DialectTable"]/tr')

            if len(rows) > 3:
                logging.debug(lxml.etree.tostring(rows[1]))
                titles = [c.xpath(r'string()').strip() for c in rows[1].xpath(r'td')]
                logging.debug(' '.join(titles))

                dialects = []
                for r in rows[2:-1]:
                    cols = r.xpath(r'td')
                    for c in cols:
                        logging.debug(lxml.etree.tostring(c))
                    dialects.append(dict(zip(titles, [c.xpath(r'string()').strip() for c in cols])))
                    logging.debug(dialects[-1])

                data['dialects'] = dialects

            print(json.dumps(data, ensure_ascii=False))

            count += 1
            if count % 100 == 0:
                logging.info('get {} characters'.format(count))

        except:
            logging.error('error get character! id = {}'.format(count), exc_info=True)

        time.sleep(0.2)

    logging.info('successfully get {} characters'.format(count)))


if __name__ == '__main__':
    main()