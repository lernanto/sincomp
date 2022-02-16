#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

'''
从原始 JSON 数据提取中古汉语信息，保存成 CSV 格式.
'''

__author__ = '黄艺华 <lernanto.wong@gmail.com>'


import sys
import logging
import json
import pandas as pd
import csv


data = []
for line in sys.stdin:
    try:
        obj = json.loads(line)
        row = {'id': obj['id'], '字形': obj['字形'].strip()}
        for k in ('攝', '聲調', '韻目', '字母', '開合', '等第', '清濁', '上字', '下字'):
            row[k] = obj['中古音'][k].strip()
        data.append(row)

    except:
        logging.error('error parsing line! {}'.format(line[:50]), exc_info=True)

data = pd.DataFrame(data)
data.index = data.pop('id')

print('# {}'.format('中古音'))
data.to_csv(sys.stdout, quoting=csv.QUOTE_NONE)
