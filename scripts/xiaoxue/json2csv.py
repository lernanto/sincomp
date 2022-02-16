#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

'''
方言读音数据 JSON 格式转 CSV 格式.
'''

__author__ = '黄艺华 <lernanto.wong@gmail.com>'


import sys
import logging
import json
import pandas as pd
import csv


dialect = None
data = []
for line in sys.stdin:
    try:
        obj = json.loads(line)
        if dialect is None:
            # 尝试根据第一条记录确定是什么方言
            # 查找记录中第一个类型为数组的元素
            for k, v in obj.items():
                if isinstance(v, list):
                    dialect = k

        if dialect in obj:
            row = {'id': obj['id']}
            for d in obj[dialect]:
                for k in ('聲母', '韻母', '調值', '調類'):
                    row['{}_{}'.format(d['方言點'].strip(), k)] = d[k].strip()

            data.append(row)

    except:
        logging.error('error passing line! {}'.format(line[:50]), exc_info=True)

data = pd.DataFrame(data)
data.index = data.pop('id')

if dialect is not None:
    print('# {}'.format(dialect))

data.to_csv(sys.stdout, quoting=csv.QUOTE_NONE)
