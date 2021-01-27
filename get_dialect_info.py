#!/usr/bin/python3 -O
# -*- encoding: utf-8 -*-

'''
从 JSON 数据提取方言点信息，保存成 CSV 格式.
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
        # 查找第一个类型为数组的元素作为方言数据
        for k, v in obj.items():
            if isinstance(v, list):
                dialect = k.strip()

                for d in v:
                    row = {'方言': dialect}
                    for k in ('區', '片', '小片', '方言點'):
                        try:
                            row[k] = d[k].strip()
                        except KeyError:
                            pass

                    data.append(row)

    except:
        logging.error('error parsing line! {}'.format(line[:50]), exc_info=True)

data = pd.DataFrame(data).drop_duplicates(['方言', '方言點']).sort_values(['方言', '區', '片', '小片', '方言點'])
data.to_csv(sys.stdout, index=False, quoting=csv.QUOTE_NONE)
