# -*- encoding: utf-8 -*-

"""
中国语言资源保护工程采录展示平台的方言数据集.

见：https://zhongguoyuyan.cn/。
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import os
import pandas
import numpy
from sklearn.neighbors import KNeighborsClassifier
import functools

from . import Dataset, clean_initial, clean_final, clean_tone


def clean_location(location):
    """
    清洗方言点地名格式.

    Parameters:
        location (`pandas.DataFrame`): 方言点信息数据表

    Returns:
        clean (`pandas.DataFrame`): 归一化市县名称的方言点数据
    """

    def norm(raw):
        clean =  raw.fillna('').str.strip() \
            .str.replace('^[(（]无[)）]$', '', regex=True) \
            .str.replace('(.)（.*）$', r'\1', regex=True) \
            .str.replace('[（）]', '', regex=True) \
            .str.replace(
                '(?:(?:土家|布依|蒙古|朝鲜|哈尼|.)族|蒙古|哈萨克|苗蔟|少数民族)*自治[州县]+$',
                '',
                regex=True
            ) \
            .str.replace('新疆生产建设兵团.+师', '', regex=True) \
            .str.replace('^(?:.*市区.*|市[内辖].+区)$', '市区', regex=True) \
            .str.replace('^(.{2,})(?:地|新|特|林|综合实验)区$', r'\1', regex=True) \
            .str.replace('(.)县城$', r'\1', regex=True) \
            .str.replace('^(.{2,6})[市州盟县区旗]$', r'\1', regex=True)

        mask = clean != raw
        if numpy.count_nonzero(mask):
            for (r, c), cnt in pandas.DataFrame({
                'raw': raw[mask],
                'clean': clean[mask]
            }).value_counts().items():
                logging.info(f'replace {r} -> {c} {cnt}')

        return clean

    clean = location.assign(
        city=norm(location['city']),
        county=norm(location['country'])
    )

    # 直辖市的地区统一置空
    mask = clean['province'].isin(['北京', '天津', '上海', '重庆'])
    clean.loc[mask & clean['county'] == '', 'county'] = clean['city']
    clean.loc[mask, 'city'] = ''

    return clean

def force_complete(data):
    """
    保证一个读音的声母、韵母、声调均有效，否则删除该读音.

    Parameters:
        data (`pandas.DataFrame`): 原始方言字音数据表

    Returns:
        output (`pandas.DataFrame`): 删除不完整读音后的方言字音数据表
    """

    columns = ['initial', 'final', 'tone']
    invalid = (data[columns].isna() | (data[columns] == '')).any(axis=1)
    invalid_num = numpy.count_nonzero(invalid)
    if invalid_num > 0:
        logging.warning(f'drop {invalid_num}/{data.shape[0]} invalid records')
        logging.warning(data[invalid])

    return data.drop(data.index[invalid])

def get_dialect(location):
    """
    从方言点信息中获取所属方言区.

    Parameters:
        location (`pandas.DataFrame`): 原始方言点信息数据表

    Returns:
        dialect (`pandas.Series`): 方言点对应的方言区列表
    """

    def clean(tag):
        """清洗原始的方言区标记."""

        tag = tag.fillna('')

        return numpy.where(
            tag.str.contains('客'),
            '客家话',
            numpy.where(
                tag.str.contains('平'),
                '平话',
                numpy.where(
                    tag.str.contains('湘南|韶州'),
                    tag.str.replace('.*(湘南|韶州).*', r'\1土话', regex=True),
                    numpy.where(
                        tag.str.contains('[吴闽赣粤湘晋徽]'),
                        tag.str.replace('.*([吴闽赣粤湘晋徽]).*', r'\1语', regex=True),
                        numpy.where(
                            tag.str.contains('北京|东北|冀鲁|胶辽|中原|兰银|江淮|西南'),
                            tag.str.replace(
                                '.*(北京|东北|冀鲁|胶辽|中原|兰银|江淮|西南).*',
                                r'\1官话',
                                regex=True
                            ),
                            ''
                        )
                    )
                )
            )
        ).astype(str)

    # 有些方言区，主要是官话的大区被标在不同的字段，尽力尝试获取
    dialect = clean(location['area'])
    dialect = numpy.where(dialect != '', dialect, clean(location['slice']))

    return pandas.Series(dialect, index=location.index)

def get_cluster(location):
    """
    从方言点信息中提取所属方言片.

    Parameters:
        location (`pandas.DataFrame`): 方言信息数据表

    Returns:
        cluster (`pandas.Series`): 方言片列表
    """

    def try_get_cluster(tag):
        """尝试从标记字符串中匹配方言片."""

        return tag[tag.str.match('^.+[^小]片.*$') == True].str.replace(
            '^(?:.*[语话]区?)?([^语话片]*[^小片]片).*$',
            r'\1',
            regex=True
        ).reindex(tag.index)

    cluster = try_get_cluster(location['slice'])
    cluster = cluster.where(cluster.notna(), try_get_cluster(location['slices']))
    cluster = cluster.where(cluster.notna(), try_get_cluster(location['area']))

    slice = location.loc[
        location['slice'].str.contains('[不未]明|[语话片]$', regex=True) == False,
        'slice'
    ]
    slice = slice.where(slice.str.len() != 2, slice + '片')
    cluster = cluster.where(
        cluster.notna(),
        slice
    )

    return cluster.fillna('')

def get_subcluster(location):
    """
    从方言点信息中提取所属方言小片.

    Parameters:
        location (`pandas.DataFrame`): 方言信息数据表

    Returns:
        subcluster (`pandas.Series`): 方言小片列表
    """

    def try_get_subcluster(tag):
        """尝试从标记字符串中匹配方言小片."""

        return tag[tag.str.match('^.+小片.*$') == True].str.replace(
            '^(?:.*[语话]区?)?(?:[^语话片]*[^小片]片)?([^语话片]+小片).*$',
            r'\1',
            regex=True
        ).reindex(tag.index)

    subcluster = try_get_subcluster(location['slices'])
    subcluster = subcluster.where(
        subcluster.notna(),
        try_get_subcluster(location['slice'])
    )
    subcluster = subcluster.where(
        subcluster.notna(),
        location.loc[
            location['slices'].str.contains('[不未]明|[语话片]$', regex=True) == False,
            'slices'
        ]
    )

    return subcluster.fillna('')


def predict_dialect(location, dialect):
    """
    使用 KNN 算法根据经纬度信息预测方言区.

    Parameters:
        location (`pandas.DataFrame`): 原始方言点信息
        dialect (`pandas.Series`): 从原始信息中提取的方言区信息，无法获取方言区的为空字符串

    Returns:
        predict (`pandas.Series`): 带预测的方言区信息，已知的方言区保持不变，其余使用 KNN 预测
    """

    mask = location['latitude'].notna() & location['longitude'].notna()
    knn = KNeighborsClassifier().fit(
        location.loc[mask & (dialect != ''), ['latitude', 'longitude']],
        dialect[mask & (dialect != '')]
    )
    predict = dialect.copy()
    predict[mask & (dialect == '')] = knn.predict(
        location.loc[mask & (dialect == ''), ['latitude', 'longitude']]
    )

    return predict


class ZhongguoyuyanDataset(Dataset):
    """
    中国语言资源保护工程采录展示平台的方言数据集.

    见：https://zhongguoyuyan.cn/。
    """

    def __init__(
        self,
        path=None,
        uniform_name=True,
        did_prefix='Z',
        cid_prefix='Z'
    ):
        """
        Parameters:
            path (str): 数据集所在的基础路径
            uniform_name (bool): 为真时，把数据列名转为通用名称
            did_prefix (str): 非空时在方言 ID 添加该前缀
            cid_prefix (str): 非空时在字 ID 添加该前缀
        """

        super().__init__('zhongguoyuyan')
        self._path = path
        self.uniform_name = uniform_name
        self.did_prefix = did_prefix
        self.cid_prefix = cid_prefix

    @property
    def path(self):
        """
        返回加载数据的基础路径，如果为空，尝试从环境变量获取路径.

        Returns:
            path (str): 数据集存放的基础路径

        如果路径为空，尝试读取环境变量 ZHONGGUOYUYAN_HONE 作为路径。
        """

        if self._path is None:
            self._path = os.environ.get('ZHONGGUOYUYAN_HOME')

        return self._path

    def load(self, *ids, variant=None, minfreq=2):
        """
        加载方言字音数据.

        Parameters:
            ids (iterable): 要加载的方言列表，当为空时，加载路径中所有方言数据
            variant (str): 要加载的方言变体，为空时加载所有变体，仅当 ids 为空时生效
            minfreq (int): 只保留每个方言中出现频次不小于该值的数据

        Returns:
            data (`pandas.DataFrame`): 方言字音表

        语保数据文件的编号由3部分组成：<方言 ID><发音人编号><内容编号>，其中：
            - 方言 ID：5个字符
            - 发音人编号：4个字符，代表老年男性、青年男性等
            - 内容编号：2个字符，dz 代表单字音
        """

        if self.path is None:
            logging.error('cannot load data due to empty path! '
                'please set environment variable ZHONGGUOYUYAN_HOME.')
            return None

        path = os.path.join(self.path, 'csv', 'dialect')
        logging.info(f'loading data from {path} ...')

        if len(ids) == 0:
            ids = sorted([e.name[:9] for e in os.scandir(path) \
                if e.is_file() and e.name.endswith('dz.csv') \
                and (variant is None or e.name[5:9] == variant)])

            if len(ids) == 0:
                logging.error(f'no data file in {path}!')
                return None

        data = []
        for id in ids:
            fname = os.path.join(path, id + 'dz.csv')
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

            # 部分音节切分错误，重新切分
            d.loc[d['initial'] == 'Ǿŋ', ['initial', 'finals']] = ['∅', 'ŋ']
            mask = d['initial'] == 'ku'
            d.loc[mask, 'initial'] = 'k'
            d.loc[mask, 'finals'] = 'u' + d.loc[mask, 'finals']

            # 部分声调被错误转为日期格式，还原成数字
            mask = d['tone'].str.match(r'^\d+年\d+月\d+日$', na='')
            d.loc[mask, 'tone'] = pandas.to_datetime(
                d.loc[mask, 'tone'],
                format=r'%Y年%m月%d日'
            ).dt.dayofyear.astype(str)

            # 清洗方言字音数据中的录入错误.
            d['initial'] = clean_initial(d['initial'])
            d['finals'] = clean_final(d['finals'])
            d['tone'] = clean_tone(d['tone'])

            if minfreq > 1:
                # 删除出现次数少的读音
                for col in ('initial', 'finals', 'tone'):
                    d.loc[
                        d.groupby(col)[col].transform('count') < minfreq,
                        col
                    ] = ''

            # 添加方言 ID 及变体 ID
            # 语保提供老年男性、青年男性等不同发音人的数据，后几个字符为其编号
            d.insert(
                0,
                'did',
                id[:5] if self.did_prefix is None else self.did_prefix + id[:5]
            )
            d.insert(1, 'variant', id[5:9])
            data.append(d)

        logging.info(f'done, {len(data)} data files loaded.')

        if len(data) == 0:
            return None

        data = pandas.concat(data, axis=0, ignore_index=True)

        if self.uniform_name:
            # 替换列名为统一的名称
            data.rename(columns={
                'iid': 'cid',
                'name': 'character',
                'finals': 'final',
                'memo': 'note'
            }, inplace=True)

            if self.cid_prefix is not None:
                # 字 ID 添加前缀
                data['cid'] = self.cid_prefix + data['cid']

        return data

    @functools.lru_cache
    def load_cache(self, *ids, variant=None):
        return self.load(*ids, variant=variant)

    def filter(self, id=None, variant=None):
        """
        筛选部分数据.

        Paramters:
            id (str or tuple): 保留的方言记录 ID
            variant (str): 保留的变体代号，仅当未指定 id 时生效

        Returns:
            data (`pandas.DataFrame`): 符合条件的数据
        """

        if id is None:
            id = []
        elif isinstance(id, str):
            id = [id]

        return self.load_cache(*id, variant=variant)

    def load_dialect_info(self, predict_dialect=False):
        """
        读取方言点信息.

        对其中的市县名称使用规则归一化，并规范化方言区名称。对缺失方言区的，使用模型预测填充。

        Parameters:
            predict_dialect (bool): 对缺失的方言区信息用机器学习模型预测填充，默认开启

        Returns:
            info (`pandas.DataFrame`): 方言点信息数据表
        """

        if self.path is None:
            logging.error('cannot load data due to empty path! '
                'please set environment variable ZHONGGUOYUYAN_HOME.')
            return None

        fname = os.path.join(self.path, 'csv', 'location.csv')
        logging.info(f'loading {self.name} dialect infomation from {fname}...')
        info = clean_location(pandas.read_csv(fname, index_col=0))

        # 以地市名加县区名为方言点名称，如地市名和县区名相同，只取其一
        info['spot'] = info['city'].where(
            info['city'] == info['county'],
            info['city'] + info['county']
        )

        # 清洗方言区、片、小片名称，需要的话预测空缺的方言区
        dialect = get_dialect(info)
        if predict_dialect:
            dialect = globals()['predict_dialect'](info, dialect)

        info['group'] = numpy.where(
            dialect.str.contains('官话'),
            '官话',
            numpy.where(dialect.str.contains('土话'), '土话', dialect)
        )
        info['subgroup'] = dialect[dialect.str.contains('官话|土话', regex=True)]
        info['subgroup'].fillna('', inplace=True)

        info['cluster'] = get_cluster(info)
        info['subcluster'] = get_subcluster(info)

        if self.did_prefix is not None:
            info.set_index(self.did_prefix + info.index, inplace=True)

        if self.uniform_name:
            info.index.rename('did', inplace=True)

        logging.info(f'done, {info.shape[0]} dialects loaded.')
        return info

    def load_char_info(self):
        """
        读取字信息.

        Returns:
            info (`pandas.DataFrame`): 字信息数据表
        """

        info = pandas.read_csv(
            os.path.join(self.path, 'csv', 'words.csv'),
            dtype=str
        )
        
        if self.cid_prefix is not None:
            info['cid'] = self.cid_prefix + info['cid']
            
        return info.set_index('cid').rename(columns={'item': 'character'})

    @functools.cached_property
    def metadata(self):
        """
        Returns:
            metadata (dict): 数据集元数据，包含：
                - dialect_info: 方言点信息
                - character_info: 字信息
        """

        return None if self.path is None else {
            'dialect_info': self.load_dialect_info(),
            'char_info': self.load_char_info()
        }

    def reset(self):
        """
        清除已经加载的数据和数据路径.
        """

        super().reset()
        self._path = None
