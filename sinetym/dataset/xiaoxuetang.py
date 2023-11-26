# -*- coding: utf-8 -*-

"""
小学堂汉字古今音资料库的现代方言数据集.

见：https://xiaoxue.iis.sinica.edu.tw/ccr。
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import os
import functools
import pandas
import opencc

from . import Dataset, clean_initial, clean_final, clean_tone


class XiaoxuetangDataset(Dataset):
    """
    小学堂汉字古今音资料库的现代方言数据集.

    见：https://xiaoxue.iis.sinica.edu.tw/ccr。
    """

    def __init__(
        self,
        path=None,
        uniform_name=True,
        did_prefix='X',
        cid_prefix='X',
        uniform_info=True
    ):
        """
        Parameters:
            path (str): 数据集所在的基础路径
            uniform_name (bool): 为真时，把数据列名转为通用名称
            did_prefix (str): 非空时在方言 ID 添加该前缀
            cid_prefix (str): 非空时在字 ID 添加该前缀
            uniform_info (bool): 小学堂原始数据为繁体中文，把方言信息转换成简体中文
        """

        super().__init__('xiaoxuetang')
        self._path = path
        self.uniform_name = uniform_name
        self.did_prefix = did_prefix
        self.cid_prefix = cid_prefix
        self.uniform_info = uniform_info

    @property
    def path(self):
        """
        返回加载数据的基础路径，如果为空，尝试从环境变量获取路径.

        Returns:
            path (str): 数据集存放的基础路径

        如果路径为空，尝试读取环境变量 XIAOXUETANG_HOME 作为路径。
        """

        if self._path is None:
            self._path = os.environ.get('XIAOXUETANG_HOME')

        return self._path

    def load(self, *ids, minfreq=2):
        """
        加载方言字音数据.

        Parameters:
            ids (iterable): 要加载的方言 ID 列表，当为空时，加载路径中所有方言数据
            minfreq (int): 只保留每个方言中出现频次不小于该值的数据

        Returns:
            data (`pandas.DataFrame`): 方言字音表
        """

        if self.path is None:
            logging.error('cannot load data due to empty path! '
                'please set environment variable XIAOXUETANG_HOME.')
            return None

        path = os.path.join(self.path, 'data', 'csv', 'dialects')
        logging.info(f'loading data from {path} ...')

        if len(ids) == 0:
            ids = []
            for e in os.scandir(path):
                id, ext = os.path.splitext(e.name)
                if e.is_file() and ext == '.csv':
                    ids.append(id)
            ids = sorted(ids)

            if len(ids) == 0:
                logging.error(f'no data file in {path}!')
                return None

        data = []
        for id in ids:
            fname = os.path.join(path, id + '.csv')
            logging.info(f'load {fname}')

            try:
                d = pandas.read_csv(
                    fname,
                    encoding='utf-8',
                    dtype=str,
                    na_filter=False
                )

            except Exception as e:
                logging.error(f'cannot load file {fname}: {e}')
                continue

            if minfreq > 1:
                # 删除出现次数少的读音
                for col in ('聲母', '韻母', '調值', '調類'):
                    d.loc[
                        d.groupby(col)[col].transform('count') < minfreq,
                        col
                    ] = ''

            d.insert(
                0,
                'did',
                id if self.did_prefix is None else self.did_prefix + id
            )
            data.append(d)

        logging.info(f'done, {len(data)} data files loaded.')

        if len(data) == 0:
            return None

        data = pandas.concat(data, axis=0, ignore_index=True)

        # 清洗方言字音数据中的录入错误
        for col, func in (
            ('聲母', clean_initial),
            ('韻母', clean_final),
            ('調值', clean_tone)
        ):
            # 一个格子可能记录了多个音，用点分隔，只取第一个
            raw = data[col].str.split('.').str[0]
            mapping = raw.value_counts().to_frame(name='count')
            mapping['clean'] = func(mapping.index)
            data[col] = raw.map(mapping['clean'])

            for i, r in mapping[mapping.index != mapping['clean']].iterrows():
                logging.info(f'{i} -> {r["clean"]} {r["count"]}')

        data['調類'] = data['調類'].str.split('.').str[0] \
            .str.replace(r'[^上中下變陰陽平去入輕聲]', '', regex=True)

        # 删除声韵调均为空的记录
        data = data[(data[['聲母', '韻母', '調值', '調類']] != '').any(axis=1)]

        if self.uniform_name:
            # 替换列名为统一的名称
            data.rename(columns={
                '字號': 'cid',
                '字': 'character',
                '聲母': 'initial',
                '韻母': 'final',
                '調值': 'tone',
                '調類': 'tone_category',
                '備註': 'note'
            }, inplace=True)

            if self.cid_prefix is not None:
                # 字 ID 添加前缀
                data['cid'] = self.cid_prefix + data['cid']

        return data

    @functools.lru_cache
    def load_cache(self, *ids):
        return self.load(*ids)

    def filter(self, id=None):
        """
        筛选部分数据.

        Paramters:
            id (str or tuple): 保留的方言 ID

        Returns:
            data (`pandas.DataFrame`): 符合条件的数据
        """

        if id is None:
            id = []
        elif isinstance(id, str):
            id = [id]

        return self.load_cache(*id)

    def load_dialect_info(self):
        """
        加载方言点信息.

        Returns:
            info (`pandas.DataFrame`): 方言点信息数据表，只包含文件中编号非空的方言点
        """

        if self.path is None:
            logging.error('cannot load data due to empty path! '
                'please set environment variable XIAOXUETANG_HOME.')
            return None

        fname = os.path.join(self.path, 'data', 'csv', 'dialect.csv')
        logging.info(f'load f{self.name} dialect information from {fname}...')
        info = pandas.read_csv(fname, dtype={'編號': str})
        info = info[info['編號'].notna()].set_index('編號')

        if self.did_prefix is not None:
            info.set_index(self.did_prefix + info.index, inplace=True)

        if self.uniform_info:
            # 把方言信息转换成简体中文
            info.update(info.select_dtypes(object).fillna('') \
                .applymap(opencc.OpenCC('t2s').convert))

            # 少数方言名称转成更通行的名称
            info['方言'].replace({'客语': '客家话', '其他土话': '土话'}, inplace=True)

        if self.uniform_name:
            info.index.rename('did', inplace=True)
            info.rename(columns={
                '方言': 'group',
                '區': 'subgroup',
                '片／小區': 'cluster',
                '小片': 'subcluster',
                '方言點': 'spot',
                '資料筆數': 'size',
                '緯度': 'latitude',
                '經度': 'longitude'
            }, inplace=True)

        logging.info(f'done, {info.shape[0]} dialects loaded.')
        return info

    @functools.cached_property
    def metadata(self):
        """
        Returns:
            metadata (dict): 数据集元数据，包含：
                - dialect_info: 方言点信息
        """

        return None if self.path is None else {
            'dialect_info': self.load_dialect_info()
        }

    def reset(self):
        """
        清除已经加载的数据和数据路径.
        """

        super().reset()
        self._path = None
