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

def get_group(location):
    """
    从方言点信息中提取所属方言区.

    Parameters:
        location (`pandas.DataFrame`): 原始方言点信息数据表

    Returns:
        group (`pandas.Series`): 方言点对应的方言区列表
    """

    def try_get_group(tag):
        """清洗原始的方言区标记."""

        tag = tag.fillna('')
        return numpy.where(
            tag.str.contains('客'),
            '客家话',
            numpy.where(
                tag.str.contains('[官平土]'),
                tag.str.replace('.*([官平土]).*', r'\1话', regex=True),
                numpy.where(
                    tag.str.contains('[吴闽赣粤湘晋徽]'),
                    tag.str.replace('.*([吴闽赣粤湘晋徽]).*', r'\1语', regex=True),
                    ''
                )
            )
        ).astype(str)

    # 有些方言区，主要是官话的大区被标在不同的字段，尽力尝试获取
    group = try_get_group(location['area'])
    group = numpy.where(group != '', group, try_get_group(location['slice']))

    return pandas.Series(group, index=location.index)

def get_subgroup(location):
    """
    从方言点信息中提取所属子分区.

    只有官话、闽语、平话、土话有子分区。

    Parameters:
        location (`pandas.DataFrame`): 原始方言点信息数据表

    Returns:
        subgroup (`pandas.Series`): 方言点对应的方言子分区列表
    """

    def try_get_subgroup(tag):
        """尝试从标记字符串中匹配方言子分区."""

        tag = tag.fillna('')
        return numpy.where(
            tag.str.contains('北京|东北|冀鲁|胶辽|中原|兰银|江淮|西南'),
            tag.str.replace('.*(北京|东北|冀鲁|胶辽|中原|兰银|江淮|西南).*', r'\1官话', regex=True),
            numpy.where(
                tag.str.contains('闽东|闽南|闽北|闽中|莆仙|邵将|琼文'),
                tag.str.replace('.*(闽东|闽南|闽北|闽中|莆仙|邵将|琼文).*', r'\1区', regex=True),
                numpy.where(
                    tag.str.contains('雷琼|琼雷'),
                    '琼文区',
                    numpy.where(
                        tag.str.contains('桂南|桂北'),
                        tag.str.replace('.*(桂南|桂北).*', r'\1平话', regex=True),
                        numpy.where(
                            tag.str.contains('湘南|粤北'),
                            tag.str.replace('.*(湘南|粤北).*', r'\1土话', regex=True),
                            numpy.where(
                                tag.str.contains('韶州|邵州'),
                                '粤北土话',
                                ''
                            )
                        )
                    )
                )
            )
        )

    subgroup = try_get_subgroup(location['slice'])
    subgroup = numpy.where(
        subgroup != '',
        subgroup,
        try_get_subgroup(location['area'])
    )

    return pandas.Series(subgroup, index=location.index)

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

def load_dialect_info(
    fname: str,
    did_prefix: str | None = None,
    uniform_name: bool = False
) -> pandas.DataFrame:
    """
    读取方言点信息

    使用规则归一化原始数据中的市县名称，以及方言大区、子区名称等信息。

    Parameters:
        fname: 方言信息文件路径
        uniform_name: 为真时，把数据列名转为通用名称
        did_prefix: 非空时在方言 ID 添加该前缀

    Returns:
        info: 方言点信息数据表
    """

    logging.info(f'loading dialect information from {fname}...')
    info = clean_location(pandas.read_csv(fname, index_col=0))

    # 以地市名加县区名为方言点名称，如地市名和县区名相同，只取其一
    info['spot'] = info['city'].where(
        info['city'] == info['county'],
        info['city'] + info['county']
    )

    # 清洗方言区、片、小片名称
    info['group'] = get_group(info)
    info['subgroup'] = get_subgroup(info)
    info['cluster'] = get_cluster(info)
    info['subcluster'] = get_subcluster(info)

    # 个别官话方言点标注的大区和子区不一致，去除
    info.loc[
        (info['group'] == '官话') & ~info['subgroup'].str.endswith('官话'),
        ['group', 'subgroup']
    ] = ''

    # 个别方言点的经纬度有误，去除
    info.loc[~info['latitude'].between(0, 55), 'latitude'] = numpy.nan
    info.loc[~info['longitude'].between(70, 140), 'longitude'] = numpy.nan

    if did_prefix is not None:
        info.set_index(did_prefix + info.index, inplace=True)

    if uniform_name:
        info.index.rename('did', inplace=True)

    logging.info(f'done, {info.shape[0]} dialects loaded.')
    return info

def load_char_info(fname, cid_prefix=None, uniform_name=False):
    """
    读取字信息.

    Parameters:
        fname (str): 字信息文件路径
        cid_prefix (str): 非空时在字 ID 添加该前缀
        uniform_name (bool): 为真时，把数据列名转为通用名称

    Returns:
        info (`pandas.DataFrame`): 字信息数据表
    """

    logging.info(f'loading character information from {fname}...')
    info = pandas.read_csv(fname, dtype=str).set_index('cid')
    
    if cid_prefix is not None:
        info.set_index(cid_prefix + info.index, inplace=True)

    if uniform_name:
        info.rename(columns={'item': 'character'}, inplace=True)

    return info
 

class ZhongguoyuyanDataset(Dataset):
    """
    中国语言资源保护工程采录展示平台的方言数据集.

    支持延迟加载。
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

        if path is None:
            raise ValueError('Empty data path.')

        super().__init__('zhongguoyuyan', metadata={
            'dialect_info': load_dialect_info(
                os.path.join(path, 'csv', 'location.csv'),
                did_prefix,
                uniform_name
            ),
            'char_info': load_char_info(
                os.path.join(path, 'csv', 'words.csv'),
                cid_prefix,
                uniform_name
            )
        })

        self.path = os.path.join(path, 'csv', 'dialect')
        self.uniform_name = uniform_name
        self.did_prefix = did_prefix
        self.cid_prefix = cid_prefix

    @functools.lru_cache
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

        logging.info(f'loading data from {self.path} ...')

        if len(ids) == 0:
            ids = sorted([e.name[:9] for e in os.scandir(self.path) \
                if e.is_file() and e.name.endswith('dz.csv') \
                and (variant is None or e.name[5:9] == variant)])

            if len(ids) == 0:
                logging.error(f'no data file in {self.path}!')
                return None

        data = []
        for id in ids:
            fname = os.path.join(self.path, id + 'dz.csv')
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

        # 清洗方言字音数据中的录入错误
        # 部分音节切分错误，重新切分
        data.loc[data['initial'] == 'Ǿŋ', ['initial', 'finals']] = ['∅', 'ŋ']
        mask = data['initial'] == 'ku'
        data.loc[mask, 'initial'] = 'k'
        data.loc[mask, 'finals'] = 'u' + data.loc[mask, 'finals']

        # 部分声调被错误转为日期格式，还原成数字
        mask = data['tone'].str.match(r'^\d+年\d+月\d+日$', na='')
        data.loc[mask, 'tone'] = pandas.to_datetime(
            data.loc[mask, 'tone'],
            format=r'%Y年%m月%d日'
        ).dt.dayofyear.astype(str)

        for col, func in (
            ('initial', clean_initial),
            ('finals', clean_final),
            ('tone', clean_tone)
        ):
            raw = data[col]
            mapping = raw.value_counts().to_frame(name='count')
            mapping['clean'] = func(mapping.index)
            data[col] = raw.map(mapping['clean'])

            for i, r in mapping[mapping.index != mapping['clean']].iterrows():
                logging.info(f'{i} -> {r["clean"]} {r["count"]}')

        # 删除声韵调均为空的记录
        data = data[(data[['initial', 'finals', 'tone']] != '').any(axis=1)]

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

    @property
    def data(self):
        return self.load()

    def filter(self, id=None, variant=None):
        """
        筛选部分数据.

        Paramters:
            id (str or tuple): 保留的方言记录 ID
            variant (str): 保留的变体代号，仅当未指定 id 时生效

        Returns:
            data (`sinetym.dataset.Dataset`): 符合条件的数据
        """

        if id is None:
            id = ()
        elif isinstance(id, str):
            id = (id,)

        return Dataset(
            self.name,
            data=self.load(*id, variant=variant),
            metadata=self.metadata
        )

    def clear_cache(self):
        """
        清除已经加载的数据.
        """

        self.load.cache_clear()
