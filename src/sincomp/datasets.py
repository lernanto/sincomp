# -*- coding: utf-8 -*-

"""
汉语方言读音数据集

当前支持读取：
    - 小学堂汉字古今音资料库的现代方言数据，见：https://xiaoxue.iis.sinica.edu.tw/ccrdata/
    - 汉字音典的现代方言数据，见：https://mcpdict.sourceforge.io/
    - 中国语言资源保护工程采录展示平台的方言数据，见：https://zhongguoyuyan.cn/
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import functools
import io
import json
import logging
import numpy
import opencc
import operator
import os
import pandas
import re
import retry
import selenium.common.exceptions
import selenium.webdriver
import selenium.webdriver.chrome.options
import selenium.webdriver.common.by
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile

from sklearn.neighbors import KNeighborsClassifier

from . import preprocess


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())


def predict_group(
    features: pandas.DataFrame | numpy.ndarray,
    labels: pandas.Series | numpy.ndarray[str]
) -> pandas.Series | numpy.ndarray[str]:
    """
    使用 KNN 算法根据经纬度信息预测方言区

    Parameters:
        features: 作为预测特征的方言点信息
        labels: 从原始信息中提取的方言区信息，无法获取方言区的为空字符串

    Returns:
        predict: 带预测的方言区信息，已知的方言区保持不变，其余使用 KNN 预测
    """

    predict = labels.copy()

    mask = numpy.all(numpy.isfinite(features), axis=1)
    predict[mask & labels.isna()] = KNeighborsClassifier().fit(
        features[mask & labels.notna()],
        labels[mask & labels.notna()]
    ).predict(features[mask & labels.isna()])

    return predict


class Dataset:
    """
    数据集基类
    """

    def __init__(
        self,
        data: pandas.DataFrame | None = None,
        name: str = 'unnamed'
    ):
        """
        Parameters:
            data: 方言字音数据表
            name: 数据集名字
        """

        self._data = None if data is None else pandas.DataFrame(data)
        self.name = str(name)

    def get_data(self, did: str) -> pandas.DataFrame:
        """
        返回指定的方言读音数据

        Parameters:
            did: 指定方言 ID

        Returns:
            data: 方言读音信息表，每行为一条记录
        """

        if self._data is None:
            raise KeyError(did)

        return self._data[self._data['did'] == did]

    @functools.cached_property
    def dialect_ids(self) -> list[str]:
        """
        返回所有方言 ID

        Returns:
            dialect_ids: 方言 ID 列表
        """

        return [] if self._data is None \
            else self._data['did'].drop_duplicates().tolist()

    @functools.cached_property
    def dialects(self) -> pandas.DataFrame:
        """
        返回数据集的方言点信息

        Returns:
            dialects: 方言信息表，每行为一个方言点，以方言 ID 为索引
        """

        return pandas.DataFrame(index=self.dialect_ids)

    @functools.cached_property
    def characters(self) -> pandas.DataFrame:
        """
        返回数据集的字信息

        Returns:
            characters: 字信息表，每行为一个字，以字 ID 为索引
        """

        data = self.data
        if 'cid' in data:
            return data.reindex(['cid', 'character'], axis=1) \
                .groupby('cid', sort=True, dropna=True).first()
        else:
            # 没有字 ID，按出现顺序编码
            return data['character'].drop_duplicates().dropna().reset_index()

    @property
    def data(self) -> pandas.DataFrame:
        """
        返回所有方言读音数据

        Returns:
            data: 合并所有方言数据的长表

        当 _data 非空时直接返回，否则依次获取所有方言点的数据，拼接成一张长表返回
        """

        if self._data is None:
            try:
                data = pandas.concat(self, axis=0, ignore_index=True)
            except ValueError:
                data = pandas.DataFrame(
                    columns=['did', 'cid', 'initial', 'final', 'tone']
                )

        else:
            data = self._data

        return data

    def items(self):
        """
        依次访问每个方言点的数据
        """

        for did in self.dialect_ids:
            yield did, self.get_data(did)

    def iterrows(self):
        """
        依次访问每各方言点的每条记录
        """

        if self._data is None:
            for data in self:
                for r in data.iterrows():
                    yield r

        else:
            yield from self._data.iterrows()

    def select(self, dids: list[str] | None = None):
        """
        从数据集中筛选方言

        Parameters:
            dids: 保留的方言 ID 列表，为空时保留全部方言

        Returns:
            output: 筛选后的数据集，只包含指定的方言
        """

        dialect_ids = self.dialect_ids
        if dids is None:
            # 保留所有方言，创建包含所有方言的新数据集是为了转换数据集类型
            dids = dialect_ids

        else:
            # 保证筛选的方言 ID 都在数据集中
            for i in dids:
                if i not in dialect_ids:
                    raise KeyError(i)

        return LinkDataset(dids, [self] * len(dids), self.name)

    def sample(self, *args, **kwargs):
        """
        从数据集随机抽样部分方言

        Parameters:
            args, kwargs: 透传给 pandas.Serires.sample 用于抽样方言

        Returns:
            output: 包含抽样方言的数据集
        """

        return self.select(self.dialects.sample(*args, **kwargs).index)

    def shuffle(
        self,
        random_state: numpy.random.RandomState | int | None = None
    ):
        """
        随机打乱数据集中方言的顺序

        Parameters:
            random_state: 用于控制打乱结果

        Returns:
            output: 内容相同的数据集，但方言的顺序随机打乱了
        """

        dids = numpy.asarray(self.dialect_ids)
        numpy.random.RandomState(random_state).shuffle(dids)
        return self.select(dids)

    def __len__(self) -> int:
        """
        返回数据集包含的方言数
        """

        return len(self.dialect_ids)

    def __iter__(self):
        return (data for _, data in self.items())

    def __getitem__(self, key):
        return self.select(key) if isinstance(key, list) else self.get_data(key)

    def __getattr__(self, name):
        if not name.startswith('_') and hasattr(pandas.DataFrame, name):
            return getattr(self.data, name)

        else:
            raise(AttributeError(
                f'{repr(type(self).__name__)} object has no attribute {repr(name)}',
                name=name,
                obj=self
            ))

    def __repr__(self):
        return f'<{type(self).__name__} {self.name} {len(self)}>'

    def __str__(self):
        if self._data is not None:
            return str(self._data)
        elif len(self) <= 1:
            return str(self.data)
        else:
            return repr(self)

    def __add__(self, other):
        """
        把另一个数据集追加到本数据集后面

        Parameters:
            other: 另一个数据集

        Returns:
            output: 新数据集，包含的方言依次为两个原始数据集的方言
        """

        bases = pandas.concat(
            [self.select()._bases, other.select()._bases],
            axis=0
        )
        return LinkDataset(bases.index, bases, 'chained')


class LinkDataset(Dataset):
    """
    从基础数据集选择指定的方言点得到的数据集

    因不保存真实的数据，只维护方言 ID 到基础数据集的链接，故名
    """

    def __init__(
        self,
        dialect_ids: list[str],
        bases: list[Dataset],
        name: str | None = None
    ):
        """
        Parameters:
            dialect_ids: 包含的方言 ID 列表
            bases: 每个方言 ID 对应的基础数据集
            name: 数据集名称
        """

        super().__init__(name=name)
        self._bases = pandas.Series(bases, index=dialect_ids)

    def get_data(self, did) -> pandas.DataFrame:
        return self._bases[did].get_data(did)

    @property
    def dialect_ids(self) -> list[str]:
        return self._bases.index.tolist()

    @functools.cached_property
    def dialects(self) -> pandas.DataFrame:
        """
        合并所有基础数据集的方言点信息，再选择本数据集的方言点
        """

        dialects = pandas.concat(
            [d.dialects for d in self._bases.drop_duplicates()],
            axis=0
        )
        return dialects[~dialects.index.duplicated(keep='first')] \
            .loc[self._bases.index]

    @functools.cached_property
    def characters(self) -> pandas.DataFrame:
        """
        合并所有基础数据据的字信息

        由于不同数据集的字编码往往不同，合并不同编码体系的字信息是未定义的
        """

        chars = pandas.concat(
            [d.characters for d in self._bases.drop_duplicates()],
            axis=0
        )
        return chars[~chars.index.duplicated(keep='first')]

    def select(self, dids: list[str] | None = None) -> Dataset:
        """
        对链接数据集的进一步操作是直接链接到基础数据集，而不是本数据集
        """

        return LinkDataset(self._bases.index, self._bases, self.name) if dids is None \
            else LinkDataset(dids, self._bases[dids], self.name)


class FileDataset(Dataset):
    """
    基于文件的数据集

    数据以 CSV 形式存放在一系列文件中，每个文件是一个方言点。
    如果指定了方言 ID 和对应的文件，则以该批文件作为数据集，否则从指定的目录查找所有文件作为数据集
    """

    def __init__(
        self,
        dialect_ids: list[str] | None = None,
        files: list[str] | None = None,
        data_dir: str | None = None,
        dialect_file: str | None = None,
        character_file: str | None = None,
        name: str | None = None
    ):
        """
        Parameters:
            dialect_ids: 包含的方言 ID 列表
            files: 每个方言 ID 对应的文件路径
            data_dir: 数据集所在的目录路径
            dialect_file: 方言信息文件路径
            character_file: 字信息文件路径
        """

        if data_dir is not None:
            data_dir = os.path.abspath(data_dir)

        if dialect_file is not None:
            dialect_file = os.path.abspath(dialect_file)
        elif data_dir is not None and \
            os.path.isfile(p := os.path.join(data_dir, '.dialects')):
            # 指定了数据目录且检测到方言信息文件
            dialect_file = p

        if character_file is not None:
            character_file = os.path.abspath(character_file)
        elif data_dir is not None and \
            os.path.isfile(p := os.path.join(data_dir, '.characters')):
            # 指定了数据目录且检测到字信息文件
            character_file = p

        if data_dir is not None:
            if dialect_ids is None:
                # 未指定方言 ID 但指定了数据目录
                # 把目录下每个文件看成一个方言点，主文件名为方言 ID
                file_map = []
                for c, _, fs in os.walk(data_dir):
                    for f in fs:
                        did = os.path.splitext(f)[0]
                        p = os.path.join(c, f)
                        if p != dialect_file and p != character_file:
                            file_map.append((did, p))

                # 按方言 ID 排序
                file_map.sort(key=operator.itemgetter(0))
                dialect_ids, files = zip(*file_map)

            elif files is None:
                # 指定了方言 ID 但未指定文件列表，以方言 ID 为文件名
                files = [os.path.join(data_dir, i) for i in dialect_ids]

        if name is None and data_dir is not None:
            name = os.path.basename(data_dir)

        super().__init__(name=name)
        self._file_map = pandas.Series(files, index=dialect_ids)
        self._dialect_file = dialect_file
        self._character_file = character_file

    @functools.cache
    def get_data(self, did: str) -> pandas.DataFrame:
        """
        加载指定方言点的数据

        Parameters:
            did: 要加载的方言 ID

        Returns:
            data: 方言读音数据表
        """

        return pandas.read_csv(self._file_map[did], dtype=str, encoding='utf-8')

    @property
    def dialect_ids(self) -> list[str]:
        return self._file_map.index.tolist()

    @functools.cached_property
    def dialects(self) -> pandas.DataFrame:
        """
        加载方言信息并返回
        """

        if self._dialect_file is None:
            return pandas.DataFrame(index=self._file_map.index)
        else:
            return pandas.read_csv(
                self._dialect_file,
                index_col='did',
                dtype={'did': str}
            ).reindex(self._file_map.index)

    @functools.cached_property
    def characters(self) -> pandas.DataFrame:
        """
        加载字信息并返回
        """

        if self._character_file is None:
            return super().characters
        else:
            return pandas.read_csv(self._character_file, dtype=str)


class CCRDataset(Dataset):
    """
    小学堂汉字古今音资料库的现代方言数据集

    见：https://xiaoxue.iis.sinica.edu.tw/ccrdata/。
    """

    def __init__(
        self,
        cache_dir: str,
        name: str = 'CCR',
        dialect_file: str = os.path.join(
            os.path.dirname(__file__),
            'ccr_dialects.csv'
        )
    ):
        """
        Parameters:
            cache_dir: 缓存文件所在目录路径
            name: 数据集名称
            dialect_file: 指定方言信息文件，默认使用随库自带的文件
        """

        super().__init__(name=name)
        self._cache_dir = os.path.abspath(cache_dir)
        self._dialect_file = os.path.abspath(dialect_file)

        # 加载方言信息
        info = pandas.read_csv(
            os.path.join(os.path.dirname(__file__), 'ccr_dialects.csv'),
            dtype=str
        ).dropna(subset=['編號'])

        # 各方言数据下载地址及下载后解压的文件路径
        self._file_map = pandas.DataFrame(
            {
                'url': 'https://xiaoxue.iis.sinica.edu.tw/ccrdata/file/' \
                    + info['方言'].map({
                    '官話': 'ccr04_guanhua_data_xlsx.zip',
                    '晉語': 'ccr05_jinyu_data_xlsx.zip',
                    '吳語': 'ccr06_wuyu_data_xlsx.zip',
                    '徽語': 'ccr07_huiyu_data_xlsx.zip',
                    '贛語': 'ccr08_ganyu_data_xlsx.zip',
                    '湘語': 'ccr09_xiangyu_data_xlsx.zip',
                    '閩語': 'ccr10_minyu_data_xlsx.zip',
                    '粵語': 'ccr11_yueyu_data_xlsx.zip',
                    '平話': 'ccr12_pinghua_data_xlsx.zip',
                    '客語': 'ccr13_keyu_data_xlsx.zip',
                    '其他土話': 'ccr14_otherdialects_data_xlsx.zip'
                }).values,
                'file': (self._cache_dir + os.sep + info['編號'] + ' ' \
                    + info['方言'].where(info['方言'] != '官話', info['區']) + \
                    '_' + info['方言點'] + '.xlsx').values
            },
            index = info['編號'].rename('did')
        )

    @classmethod
    def clean_subgroup(cls, subgroup: pandas.Series) -> pandas.Series:
        """
        清洗方言子分区信息

        只有官话、闽语、平话、土话有子分区。

        Parameters:
            subgroup: 原始方言子分区信息列表

        Returns:
            output: 清洗后的方言子分区列表
        """

        return pandas.Series(numpy.where(
            subgroup.str.contains('北京|東北|冀鲁|胶遼|中原|蘭銀|江淮|西南', na=False),
            subgroup.str.replace(
                '.*(北京|東北|冀鲁|胶遼|中原|蘭銀|江淮|西南).*',
                r'\1官話',
                regex=True
            ),
            numpy.where(
                subgroup.str.contains('閩東|閩南|閩北|閩中|莆仙|邵將|瓊文', na=False),
                subgroup.str.replace(
                    '.*(閩東|閩南|閩北|閩中|莆仙|邵將|瓊文).*',
                    r'\1區',
                    regex=True
                ),
                numpy.where(
                    subgroup.str.contains('^桂南|桂北', na=False),
                    subgroup.str.replace('^(桂南|桂北).*', r'\1平話', regex=True),
                    numpy.where(
                        subgroup.str.contains('湘南|粵北', na=False),
                        subgroup.str.replace('.*(湘南|粵北).*', r'\1土話', regex=True),
                        ''
                    )
                )
            )
        ), index=subgroup.index).replace('', pandas.NA)

    @staticmethod
    @retry.retry(exceptions=urllib.error.URLError, tries=3, delay=1)
    def download(url: str, output: str) -> None:
        """
        从小学堂网站下载方言读音数据

        Parameters:
            url: 下载地址
            output: 保存下载解压文件的本地目录
        """

        logger.info(f'downloading {url} ...')

        # 设置 User-Agent，否则请求会被拒绝
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as res:
            with zipfile.ZipFile(io.BytesIO(res.read())) as zf:
                logger.info(f'extracting files to {output} ...')
                os.makedirs(output, exist_ok=True)

                for info in zf.infolist():
                    # 压缩包路径编码为 Big5，但 zipfile 默认用 CP437 解码，需重新用 Big5 解码
                    try:
                        fname = info.filename.encode('cp437').decode('big5')
                    except UnicodeError:
                        fname = info.filename
                    # 改正文件名中的别字
                    fname = fname.replace('閔', '閩')

                    logger.info(f'extracting {fname} ...')
                    with open(os.path.join(output, fname), 'wb') as of:
                        of.write(zf.read(info))

        logger.info('done.')

    @functools.cache
    def get_data(self, did: str) -> pandas.DataFrame:
        """
        加载方言字音数据

        Parameters:
            did: 要加载的方言 ID

        Returns:
            data: 方言读音数据表

        如果要加载的数据文件不存在，先从网站下载。
        """

        info = self._file_map.loc[did]
        if not os.path.isfile(info['file']):
            # 方言数据文件不存在，从网站下载
            self.download(info['url'], self._cache_dir)

        logger.info(f'loading data from {info["file"]} ...')
        data = pandas.read_excel(info['file'], dtype=str)

        # 替换列名为统一的名称
        data.rename(columns={
            '字號': 'cid',
            '字': 'character',
            '聲母': 'initial',
            '韻母': 'final',
            '調值': 'tone',
            '調類': 'tone_category',
            '備註': 'note',
            'Order': 'cid',
            'Char': 'character',
            'ShengMu': 'initial',
            'YunMu': 'final',
            'DiaoZhi': 'tone',
            'DiaoLei': 'tone_category',
            'Comment': 'note'
        }, inplace=True)

        # 多音字每个读音为一行，但有些多音字声韵调部分相同的，只有其中一行标了数据，
        # 其他行为空。对于这些空缺，使用同字第一个非空的的读音填充
        data.fillna(
            data.groupby('cid')[['initial', 'final', 'tone', 'tone_category']].transform('first'),
            inplace=True
        )

        # 清洗数据集特有的错误
        if id == '118':
            data['final'] = data['final'].str.translate({
                0x003f: 0x028f, # QUESTION MARK -> LATIN LETTER SMALL CAPITAL Y
            })
        elif id == '178':
            data['initial'] = data['initial'].str.translate({
                0x0237: 0x0255, # LATIN SMALL LETTER DOTLESS J -> LATIN SMALL LETTER C WITH CURL
            })

        # 清洗读音数据。一个格子可能记录了多个音，用点分隔，只取第一个
        data['initial'] = preprocess.clean_initial(
            preprocess.clean_ipa(data['initial'].str.split('.').str[0])
        )
        data['final'] = preprocess.clean_final(
            preprocess.clean_ipa(data['final'].str.split('.').str[0])
        )
        data['tone'] = preprocess.clean_tone(
            data['tone'].str.split('.').str[0].str.translate({
                0x0030: 0x2205, # DIGIT ZERO -> EMPTY SET
            })
        )
        data['tone_category'] = data['tone_category'].str.split('.').str[0] \
            .str.replace(r'[^上中下變陰陽平去入輕聲]', '', regex=True)

        # 删除声韵调均为空的记录
        data.replace('', pandas.NA, inplace=True)
        data.dropna(how='all', subset=['initial', 'final', 'tone'], inplace=True)

        data['did'] = did
        return data[[
            'did',
            'cid',
            'character',
            'initial',
            'final',
            'tone',
            'tone_category',
            'note'
        ]]

    @property
    def dialect_ids(self) -> list[str]:
        return self.dialects.index.tolist()

    @functools.cached_property
    def dialects(self) -> pandas.DataFrame:
        """
        加载方言点信息并返回
        """

        dialects = pandas.read_csv( self._dialect_file, dtype={'編號': str}) \
            .rename(columns={
                '方言': 'group',
                '區': 'subgroup',
                '片／小區': 'cluster',
                '小片': 'subcluster',
                '方言點': 'name',
                '緯度': 'latitude',
                '經度': 'longitude',
                '編號': 'did'
            }) \
            .dropna(subset=['did']).set_index('did')

        # 少数方言名称转成更通行的名称；部分方言点包含来源文献，删除
        dialects = dialects.replace({'group': {'客語': '客家話', '其他土話': '土話'}}) \
            .assign(subgroup=self.clean_subgroup(dialects['subgroup'])) \
            .assign(spot=dialects['name'].str.replace(
                r'\(安徽省志\)|\(珠江三角洲\)|\(客贛方言調查報告\)|\(廣西漢語方言\)'
                r'|\(平話音韻研究\)|\(廣東閩方言\)|\(漢語方音字匯\)|\(當代吳語\)',
                '',
                regex=True
            ))

        # 把方言信息转换成简体中文
        dialects.update(
            dialects[['group', 'subgroup', 'cluster', 'subcluster', 'spot']] \
                .map(opencc.OpenCC('t2s').convert, na_action='ignore')
        )

        return dialects.reindex([
            'name',
            'province',
            'city',
            'county',
            'town',
            'village',
            'group',
            'subgroup',
            'cluster',
            'subcluster',
            'spot',
            'latitude',
            'longitude'
        ], axis=1)

    @functools.cached_property
    def characters(self) -> pandas.DataFrame:
        """
        加载字信息

        优先从缓存文件加载，如果文件不存在，会根据全部方言数据统计，这样会触发下载全部方言
        """

        try:
            # 尝试从缓存文件加载字信息
            path = os.path.join(self._cache_dir, '.characters')
            logger.debug(f'load character information from {path}')
            return pandas.read_csv(
                path,
                dtype=str,
                encoding='utf-8'
            ).set_index('cid')

        except FileNotFoundError:
            # 缓存文件不存在，从方言读音数据统计字信息
            logger.info(f'{path} not found, get dialect information from data.')
            characters = super().characters

            # 保存到缓存文件
            logger.debug(f'save dialect information to {path}')
            characters.to_csv(path, encoding='utf-8', lineterminator='\n')

            return characters


class MCPDictDataset(Dataset):
    """
    汉字音典方言数据集

    见：https://mcpdict.sourceforge.io/。
    """

    def __init__(
        self,
        cache_dir: str,
        empty: str | None = '∅',
        name: str = 'MCPDict'
    ):
        """
        Parameters:
            cache_dir: 缓存文件所在目录路径
            empty: 代表零声母/零韵母/零声调的字符串，为 None 时保持原状
            name: 数据集名称
        """

        super().__init__(name=name)
        self._cache_dir = os.path.abspath(cache_dir)
        self._empty = empty

    @staticmethod
    @retry.retry(exceptions=urllib.error.URLError, tries=3, delay=1)
    def download(
        output: str,
        url: str = 'https://github.com/osfans/MCPDict/archive/refs/heads/master.zip'
    ) -> None:
        """
        从 MCPDict 项目主页下载数据

        Parameters:
            output: 保存下载解压文件的本地目录
            url: 项目下载地址
        """

        logger.info(f'downloading {url} ...')

        with urllib.request.urlopen(url) as res:
            with zipfile.ZipFile(io.BytesIO(res.read())) as zf:
                os.makedirs(
                    os.path.join(output, 'tools', 'tables', 'output'),
                    exist_ok=True
                )
                logger.info(f'extracting files to {output} ...')

                for info in zf.infolist():
                    # 路径第一段是带版本号的项目名，需去除
                    path = info.filename.partition('/')[2]
                    # 把字音数据目录的所有文件解压到目标路径
                    if not info.is_dir() and path.startswith('tools/tables/output/'):
                        logger.info(f'extracting {info.filename} ...')
                        path = os.path.join(*[output] + path.split('/'))
                        with open(path, 'wb') as of:
                            of.write(zf.read(info))

        logger.info('done.')

    @functools.cached_property
    def tone_map(self) -> dict[str, tuple[dict[str, str], dict[str, str]]]:
        """
        从方言详情提取声调调值和调类的映射表

        Returns:
            tone_map: 方言 ID 到声调映射表的映射表，其值又是映射表的二元组，
                前者为调号到调值的映射表，后者为调号到调类的映射表

        如果文件不存在，先从项目页面下载。
        """

        path = os.path.join(self._cache_dir, 'tools', 'tables', 'output')
        if not os.path.isdir(path):
            # 数据文件不存在，先从汉字音典项目页面下载
            self.download(self._cache_dir)

        fname = os.path.join(path, '_詳情.json')
        info = pandas.read_json(fname, orient='index', encoding='utf-8')

        tm = {}
        for i, m in info['聲調'].map(json.loads).items():
            tone = {}
            cat = {}
            for k, v in m.items():
                # 少数连读声调有特殊符号，暂时去除
                tone[k] = re.sub(f'[^{"".join(preprocess._TONES)}]', '', v[0])
                cat[k] = v[3]
            tm[i] = tone, cat

        return tm

    @functools.cache
    def get_data(self, did: str) -> pandas.DataFrame:
        """
        加载方言读音数据

        Parameters:
            did: 要加载的方言 ID

        Returns:
            data: 方言读音数据表

        如果文件不存在，先从项目页面下载。
        """

        path = os.path.join(self._cache_dir, 'tools', 'tables', 'output')
        if not os.path.isdir(path):
            # 数据文件不存在，先从汉字音典项目页面下载
            self.download(self._cache_dir)

        fname = os.path.join(path, did + '.tsv')
        logger.debug(f'load data from {fname}')
        data = pandas.read_csv(
            fname,
            sep='\t',
            header=None,
            names=['character', 'ipa', 'note'],
            dtype=str,
            na_values={'character': '\u25a1'},   # 方框代表有音无字
            comment='#',
            encoding='utf-8'
        )

        data['ipa'] = data['ipa'].str.translate({
            0x008f: 0x027f, # -> LATIN SMALL LETTER REVERSED R WITH FISHHOOK
            0x0090: 0x0285, # -> LATIN SMALL LETTER SQUAT REVERSED ESH
        })
        data.replace('', pandas.NA, inplace=True)

        # 把原始读音切分成声母、韵母、声调
        seg = data.pop('ipa').str.extract(r'([^0-9]*)([0-9][0-9a-z]*)?')
        data[['initial', 'final']] = preprocess.parse(
            preprocess.clean_ipa(seg.iloc[:, 0], force=True)
        ).iloc[:, :2]

        # 汉字音典的原始读音标注的是调号，根据方言详情映射成调值和调类
        tone, cat = self.tone_map[did]
        data['tone'] = seg.iloc[:, 1].map(tone)
        data['tone_category'] = seg.iloc[:, 1].map(cat)

        # 删除声韵调均为空的记录
        data.dropna(
            how='all',
            subset=['initial', 'final', 'tone'],
            inplace=True
        )

        # 根据需要替换空值
        if self._empty is not None:
            data.replace(
                {'initial': '', 'final': '', 'tone': ''},
                self._empty,
                inplace=True
            )

        data['did'] = did
        return data[[
            'did',
            'character',
            'initial',
            'final',
            'tone',
            'tone_category',
            'note'
        ]]

    @property
    def dialect_ids(self) -> list[str]:
        return self.dialects.index.tolist()

    @functools.cached_property
    def dialects(self) -> pandas.DataFrame:
        """
        加载方言点信息并返回，如果文件不存在，先从项目页面下载。
        """

        path = os.path.join(self._cache_dir, 'tools', 'tables', 'output')
        if not os.path.isdir(path):
            # 数据文件不存在，先从汉字音典项目页面下载
            self.download(self._cache_dir)

        fname = os.path.join(path, '_詳情.json')
        logger.debug(f'load dialect information from {fname}')
        dialects = pandas.read_json(fname, orient='index', encoding='utf-8')

        # 汉典的方言数据实际来自小学堂，已收录在小学堂数据集，此处剔除
        # 汉字音典数据包含历史拟音、域外方音和一些拼音方案，只取用国际音标注音的现代方言数据
        dialects = dialects[
            (dialects['文件格式'] != '漢典') \
            & (~dialects['地圖集二分區'].isin(['歷史音', '現代標準漢語', '民族語', '域外方音', '戲劇'])) \
            & (~dialects.index.str.match('^1[0-9]{3}') | dialects.index.isin([
                '1935永明',
                '1935南昌',
                '1935醴陵',
                '1935長沙',
            ])) \
            & (~dialects.index.isin([
                '鄕音字類',
                '淸末寧波',
                '淸末溫州',
                '訓詁諧音',
                '湘音檢字',
                '香港',
                '臺灣'
            ])) \
            & ((path + os.sep + dialects.index + '.tsv').map(os.path.isfile))
        ]

        # 解析方言分类
        cat = dialects['地圖集二分區'].str.split('-')
        # 乡话使用了异体字，OpenCC 无法转成简体，特殊处理
        dialects = dialects.assign(
            group=cat.str[0].replace('鄕話', '鄉話'),
            cluster=cat.str[1],
            subcluster=cat.str[2]
        )

        mask = dialects['group'].str.endswith('官話') \
            | dialects['group'].str.endswith('官话')
        dialects.loc[mask, 'subgroup'] = dialects.loc[mask, 'group']
        dialects.loc[mask, 'group'] = '官話'

        # 原始分区不分平话和土话，根据子分区信息尽量分开
        mask = dialects['group'] == '平話和土話'
        dialects.loc[mask, 'group'] = numpy.where(
            dialects.loc[mask, 'cluster'].isin(['桂南片', '桂北片']),
            '平話',
            '土話'
        )

        # 解析经纬度
        dialects[['latitude', 'longitude']] = dialects['經緯度'].str.partition(',') \
            .iloc[:, [2, 0]].astype(float, errors='ignore')

        dialects = dialects.rename_axis('did').rename(columns={
            '語言': 'name',
            '簡稱': 'spot',
            '省': 'province',
            '市': 'city',
            '縣': 'county',
            '鎮': 'town',
            '村': 'village',
        })

        # 把方言信息转换成简体中文
        dialects.update(dialects[[
            'province',
            'city',
            'county',
            'town',
            'village',
            'group',
            'subgroup',
            'cluster',
            'subcluster',
            'spot'
        ]].map(opencc.OpenCC('t2s').convert, na_action='ignore'))

        return dialects.reindex([
            'name',
            'province',
            'city',
            'county',
            'town',
            'village',
            'group',
            'subgroup',
            'cluster',
            'subcluster',
            'spot',
            'latitude',
            'longitude'
        ], axis=1)

    @functools.cached_property
    def characters(self) -> pandas.DataFrame:
        """
        加载字信息

        优先从缓存文件加载，如果文件不存在，会根据全部方言数据统计，这样会触发下载全部方言
        """

        try:
            # 尝试从缓存文件加载字信息
            path = os.path.join(self._cache_dir, '.characters')
            logger.debug(f'load character information from {path}')
            return pandas.read_csv(
                path,
                dtype=str,
                encoding='utf-8'
            ).set_index('cid')

        except FileNotFoundError:
            # 缓存文件不存在，从方言读音数据统计字信息
            logger.info(f'{path} not found, get dialect information from data.')
            characters = super().characters

            # 保存到缓存文件
            logger.debug(f'save dialect information to {path}')
            characters.to_csv(path, encoding='utf-8', lineterminator='\n')

            return characters


class ZhongguoyuyanDownloader:
    """
    用于从中国语言资源保护工程采录展示平台网站下载方言数据的工具类
    """

    def __init__(
        self,
        url: str = 'https://zhongguoyuyan.cn',
        timeout: float = 300,
        delay: float = 10
    ):
        """
        Parameters:
            url: 中国语言资源保护工程采录展示平台的网址
            timeout: 请求网页的超时时间
            delay: 请求之间间隔的秒数，为减轻网站压力，早于间隔时间的请求会等够间隔时间才真正请求
        """

        self._url = url
        self._timeout = timeout
        self._delay = delay
        self._lock = threading.Lock()
        self._last_get_time = None
        self._get_count = 0

    def __del__(self) -> None:
        # 关闭浏览器
        logger.info('close browser.')
        try:
            self._driver.quit()
        except (
            AttributeError,
            selenium.common.exceptions.InvalidSessionIdException
        ):
            ...

    def get(
        self,
        path: str,
        selector: str | None = None,
        data_path: str | None = None
    ) -> str:
        """
        使用外部浏览器从网站下载指定的数据

        Parameters:
            path: 数据对应的网页路径
            selector: 使用 CSS 选择器指定元素，等待该元素就绪再取数据
            data_path: 实际数据资源的路径，用于提取数据，为空时和 `path` 相同

        通过调用浏览器请求数据所在的页面实现，如该页面要求登录访问，需在浏览器界面手工登录，
        然后自动跳转到请求的页面
        """

        url = self._url + path
        data_url = url if data_path is None else self._url + data_path

        # 同一时间只允许一个线程操作浏览器
        with self._lock:
            if self._delay is not None and self._last_get_time is not None:
                intv = time.time() - self._last_get_time
                if intv < self._delay:
                    # 请求间隔时间过短，等够间隔时间
                    left = self._delay - intv
                    logger.info(
                        f'{intv:.0f}s < {self._delay:.0f}s since last request, '
                        f'please wait another {left:.0f}s.'
                    )
                    time.sleep(left)

            try:
                self._driver.title
            except (
                AttributeError,
                selenium.common.exceptions.InvalidSessionIdException
            ):
                # 未打开浏览器，或会话已过期，重新打开浏览器，开启日志以便跟踪请求到的数据
                options = selenium.webdriver.chrome.options.Options()
                options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
                logger.info(f'open Chrome with options = {options} .')
                self._driver = selenium.webdriver.Chrome(options=options)
                # 设置等待超时时间
                self._driver.implicitly_wait(self._timeout)

            # 调用浏览器请求指定数据，如需登录，需在浏览器界面手工登录，然后会自动跳转到请求的页面
            logger.debug(f'get {url}')
            self._last_get_time = time.time()
            self._driver.get(url)
            self._get_count += 1
            if self._get_count % 100 == 0:
                logger.warning(
                    f'{self._get_count} requests reached, '
                    'consider pause or your account may be BANNED!'
                )

            # 等待指定的网页元素加载完毕
            if selector is not None:
                logger.debug(f'wait for element {selector}')
                self._driver.find_element(
                    selenium.webdriver.common.by.By.CSS_SELECTOR,
                    selector
                )

            # 从浏览器网络日志查找后台真正的数据资源信息
            logger.debug(f'search for response of {data_url}')
            request_id = None
            for log in self._driver.get_log('performance'):
                try:
                    message = json.loads(log['message'])['message']
                    if message['method'] == 'Network.responseReceived' \
                        and message['params']['response']['url'] == data_url:
                            request_id = message['params']['requestId']
                            logger.debug(f'found {data_url}, requestId = {request_id}')

                except (KeyError, TypeError) as e:
                    logger.warning(e)

            if request_id is None:
                raise RuntimeError(f'no response received for {data_url} !')
            else:
                # 调用浏览器获取响应数据
                rsp = self._driver.execute_cdp_cmd(
                    'Network.getResponseBody',
                    {'requestId': request_id}
                )
                return rsp.get('body')

    def get_survey(self) -> str:
        """
        获取调查的所有方言点信息

        Returns:
            data: 从网站下载的方言点信息原始数据
        """

        return self.get(
            '/index',
            'span.number___n9qo2',
            '/api/mongo/query/latestSurveyMongo'
        )

    def get_standard(self) -> str:
        """
        获取方言调查标准，包括字、词、句的列表

        Returns:
            data: 从网站下载的方言调查标准原始数据
        """

        return self.get('/api/api/media/standard')

    def get_point(self, id: str) -> str:
        """
        获取指定的方言点数据

        Parameters:
            id: 方言 ID

        Returns:
            data: 从网站下载的方言点原始数据
        """

        return self.get(
            '/point/' + id,
            '[data-row-key="0001"]',
            '/api/mongo/resource/normal'
        )


class ZhongguoyuyanDataset(Dataset):
    """
    中国语言资源保护工程采录展示平台的方言数据集

    见：https://zhongguoyuyan.cn/。
    """

    def __init__(
        self,
        cache_dir: str,
        name: str = 'zhongguoyuyan',
        downloader_kwargs: dict = {}
    ):
        """
        Parameters:
            cache_dir: 缓存文件所在目录路径
            name: 数据集名称
            downloader_kwargs: 传给下载器的参数
        """

        super().__init__(name=name)
        self._cache_dir = cache_dir
        self._downloader = ZhongguoyuyanDownloader(**downloader_kwargs)

    def load_or_download(self, name: str):
        """
        从本地缓存文件加载数据，如不存在本地缓存文件，先从网站下载
        """

        path = os.path.join(self._cache_dir, name + '.json')
        try:
            with open(path, encoding='utf-8') as f:
                # 存在本地缓存文件，从本地加载
                logger.debug(f'load cache file {path}')
                return json.load(f)

        except FileNotFoundError:
            # 不存在本地缓存文件，从网站下载
            logger.info(f'cache file {path} not existing, download from the Web.')
            if name == 'survey':
                # 所有调查的方言点信息
                raw = self._downloader.get_survey()
                data = json.loads(raw)

            elif name == 'standard':
                # 方言调查标准
                raw = self._downloader.get_standard()
                data = json.loads(raw)

            else:
                # 某个方言点的数据
                raw = self._downloader.get_point(name)
                data = json.loads(raw)
                if data['code'] != 200:
                    # 获取方言点数据失败，不写缓存
                    raise RuntimeError(data['description'])

            # 保存缓存文件
            logger.debug(f'save cache file {path}')
            os.makedirs(self._cache_dir, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(raw)

            return data

    @staticmethod
    def clean_location(raw: pandas.Series) -> pandas.Series:
        return raw.str.strip() \
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

    @classmethod
    def get_group(cls, location: pandas.DataFrame) -> pandas.Series:
        """
        从方言点信息中提取所属方言区

        Parameters:
            location: 原始方言点信息数据表

        Returns:
            group: 方言点对应的方言区列表
        """

        def try_get_group(tag: pandas.Series) -> pandas.Series:
            """清洗原始的方言区标记"""

            return pandas.Series(numpy.where(
                tag.str.contains('客', na=False, regex=False),
                '客家话',
                numpy.where(
                    tag.str.contains('[官平土]', na=False),
                    tag.str.replace('.*([官平土]).*', r'\1话', regex=True),
                    numpy.where(
                        tag.str.contains('[吴闽赣粤湘晋徽]', na=False),
                        tag.str.replace('.*([吴闽赣粤湘晋徽]).*', r'\1语', regex=True),
                        ''
                    )
                )
            ), tag.index)

        # 有些方言区，主要是官话的大区被标在不同的字段，尽力尝试获取
        group = try_get_group(location['area'])
        group.where(group != '', try_get_group(location['slice']), inplace=True)

        # 对平话和土话的标注不一致，尽量和下面的子分区对齐
        group[
            (group == '土话') \
            & location['slice'].str.contains('桂南|桂北', na=False, regex=True)
        ] = '平话'

        return group.replace('', pandas.NA)

    @classmethod
    def get_subgroup(self, location: pandas.DataFrame) -> pandas.Series:
        """
        从方言点信息中提取所属子分区

        只有官话、闽语、平话、土话有子分区。

        Parameters:
            location: 原始方言点信息数据表

        Returns:
            subgroup: 方言点对应的方言子分区列表
        """

        def try_get_subgroup(tag: pandas.Series) -> pandas.Series:
            """尝试从标记字符串中匹配方言子分区"""

            return pandas.Series(numpy.where(
                tag.str.contains('北京|东北|冀鲁|胶辽|中原|兰银|江淮|西南', na=False),
                tag.str.replace(
                    '.*(北京|东北|冀鲁|胶辽|中原|兰银|江淮|西南).*',
                    r'\1官话',
                    regex=True
                ),
                numpy.where(
                    tag.str.contains('闽东|闽南|闽北|闽中|莆仙|邵将|琼文', na=False),
                    tag.str.replace(
                        '.*(闽东|闽南|闽北|闽中|莆仙|邵将|琼文).*',
                        r'\1区',
                        regex=True
                    ),
                    numpy.where(
                        tag.str.contains('雷琼|琼雷', na=False),
                        '琼文区',
                        numpy.where(
                            tag.str.contains('桂南|桂北', na=False),
                            tag.str.replace('.*(桂南|桂北).*', r'\1平话', regex=True),
                            numpy.where(
                                tag.str.contains('湘南|粤北', na=False),
                                tag.str.replace(
                                    '.*(湘南|粤北).*',
                                    r'\1土话',
                                    regex=True
                                ),
                                numpy.where(
                                    tag.str.contains('韶州|邵州', na=False),
                                    '粤北土话',
                                    ''
                                )
                            )
                        )
                    )
                )
            ), tag.index)

        subgroup = try_get_subgroup(location['slice'])
        subgroup.where(
            subgroup != '',
            try_get_subgroup(location['area']),
            inplace=True
        )

        return subgroup.replace('', pandas.NA)

    @classmethod
    def get_cluster(self, location: pandas.DataFrame) -> pandas.Series:
        """
        从方言点信息中提取所属方言片

        Parameters:
            location: 方言信息数据表

        Returns:
            cluster: 方言片列表
        """

        def try_get_cluster(tag: pandas.Series) -> pandas.Series:
            """尝试从标记字符串中匹配方言片"""

            return tag[tag.str.match('^.+[^小]片.*$') == True].str.replace(
                '^(?:.*[语话]区?)?([^语话片]*[^小片]片).*$',
                r'\1',
                regex=True
            ).reindex(tag.index)

        cluster = try_get_cluster(location['slice'])
        cluster.where(
            cluster.notna(),
            try_get_cluster(location['slices']),
            inplace=True
        )
        cluster.where(
            cluster.notna(),
            try_get_cluster(location['area']),
            inplace=True
        )

        slice = location.loc[
            location['slice'].str.contains('[不未]明|[语话片]$', regex=True) == False,
            'slice'
        ]
        cluster.where(
            cluster.notna(),
            slice.where(slice.str.len() != 2, slice + '片'),
            inplace=True
        )

        return cluster

    @classmethod
    def get_subcluster(self, location: pandas.DataFrame) -> pandas.Series:
        """
        从方言点信息中提取所属方言小片

        Parameters:
            location: 方言信息数据表

        Returns:
            subcluster: 方言小片列表
        """

        def try_get_subcluster(tag: pandas.Series) -> pandas.Series:
            """尝试从标记字符串中匹配方言小片"""

            return tag[tag.str.match('^.+小片.*$') == True].str.replace(
                '^(?:.*[语话]区?)?(?:[^语话片]*[^小片]片)?([^语话片]+小片).*$',
                r'\1',
                regex=True
            ).reindex(tag.index)

        subcluster = try_get_subcluster(location['slices'])
        subcluster.where(
            subcluster.notna(),
            try_get_subcluster(location['slice']),
            inplace=True
        )
        subcluster.where(
            subcluster.notna(),
            location.loc[
                location['slices'].str.contains(
                    '[不未]明|[语话片]$',
                    regex=True
                ) == False,
                'slices'
            ],
            inplace=True
        )

        return subcluster

    def get_dialects(self, refresh: bool = False) -> pandas.DataFrame:
        """
        从每个方言文件读取方言信息

        Parameters:
            refresh: 强制重新生成缓存文件

        Returns:
            dialects: 方言点信息数据表

        从已下载的文件读取方言信息，不下载新文件。并生成缓存文件。当下载了新方言文件后，
        需强制刷新才能得到新下载方言的信息。
        """

        path = os.path.join(self._cache_dir, '.dialects')
        if os.path.isfile(path) and not refresh:
            # 缓存文件已存在且不强制刷新，从缓存文件读取方言信息
            logger.info(f'load dialect information from cache file {path} .')
            dialects = pandas.read_csv(path, encoding='utf-8', dtype={'did': str}) \
                .set_index('did')

        else:
            # 缓存文件不存在或强制刷新，从每个方言文件读取
            dialects = []
            for did in self.dialect_ids:
                fname = os.path.join(self._cache_dir, did + '.json')
                try:
                    with open(fname, encoding='utf-8') as f:
                        data = json.load(f)
                        loc = data['data']['mapLocation']
                        dialects.append({**loc['location'], **loc['point']})

                except FileNotFoundError:
                    logger.debug(
                        f'{path} not found, dialect information absent for {did}'
                    )

            if len(dialects) == 0:
                # 未下载任何方言文件，方言信息为空
                dialects = pandas.DataFrame()

            else:
                dialects = pandas.DataFrame(dialects) \
                    .rename(columns={'firstLevelid': 'did', 'country': 'county'}) \
                    .set_index('did') \
                    .replace(['', '(无)', '(无）', '无', '（无)', '（无）'], pandas.NA)

                # 从文件名提取方言名称
                dialects['name'] = dialects['filepath'].str.extract(r'/([^#]+)#?需交文件电子版')

                # 以县区名为方言点名称，如县区名为空，依次上溯市名、省级行政区名
                dialects['spot'] = dialects['county'].where(
                    dialects['county'].notna(),
                    dialects['city'].where(dialects['city'].notna(), dialects['province'])
                )

                # 清洗方言区、片、小片名称
                dialects['group'] = self.get_group(dialects)
                dialects['subgroup'] = self.get_subgroup(dialects)
                dialects['cluster'] = self.get_cluster(dialects)
                dialects['subcluster'] = self.get_subcluster(dialects)

                # 个别官话方言点标注的大区和子区不一致，去除
                dialects.loc[
                    (dialects['group'] == '官话') \
                        & ~dialects['subgroup'].str.endswith('官话', na=False),
                    ['group', 'subgroup']
                ] = pandas.NA

                # 个别方言点的经纬度有误，去除
                dialects.loc[~dialects['latitude'].between(0, 55), 'latitude'] = numpy.nan
                dialects.loc[~dialects['longitude'].between(70, 140), 'longitude'] = numpy.nan

            dialect_ids = self.dialect_ids
            if dialects.shape[0] < len(dialect_ids):
                logger.warning(
                    f'generate cache for only {dialects.shape[0]}/{len(dialect_ids)} '
                    'dialect(s), after you download new dialect files, '
                    'call get_dialects(refresh=True) to refresh cache.'
                )

            dialects = dialects.reindex([
                'name',
                'province',
                'city',
                'county',
                'town',
                'village',
                'group',
                'subgroup',
                'cluster',
                'subcluster',
                'spot',
                'latitude',
                'longitude'
            ], axis=1).reindex(dialect_ids)

            # 保存到缓存文件
            logger.info(f'save dialect information to cache file {path} .')
            dialects.to_csv(path, encoding='utf-8', lineterminator='\n')

        return dialects

    @functools.cache
    def get_data(self, did: str) -> pandas.DataFrame:
        """
        加载方言字音数据

        Parameters:
            did: 要加载的方言 ID

        Returns:
            data: 方言字音表

        语保数据文件包含了单字、词汇和语法，其中单字包含了老年男性和青年男性两个发音人的数据，
        当前只取其中单字的老年男性数据。如果文件不存在，先从网站下载。
        """

        point = self.load_or_download(did)
        # 替换列名为统一的名称
        data = pandas.json_normalize(
            point['data']['resourceList'][0]['items'], 
            'records',
            ['iid', 'name']
        ).rename(columns={
            'iid': 'cid',
            'name': 'character',
            'finals': 'final',
            'memo': 'note'
        })

        # 清洗数据集特有的错误
        if id == '02135':
            data['final'] = data['final'].str.translate({
                0xf175: 0x0303, # -> COMBINING TILDE
                0xf179: 0x0303, # -> COMBINING TILDE
            })

        # 部分声调被错误转为日期格式，还原成数字
        mask = data['tone'].str.fullmatch(r'\d+年\d+月\d+日', na=False)
        data.loc[mask, 'tone'] = pandas.to_datetime(
            data.loc[mask, 'tone'],
            format=r'%Y年%m月%d日'
        ).dt.dayofyear.astype(str)

        # 个别声调被错误转成浮点数
        data['tone'] = data['tone'].str.replace(r'\.0$', '', regex=True)

        # 清洗读音 IPA
        data['initial'] = preprocess.clean_initial(
            preprocess.clean_ipa(data['initial']).str.translate({
                0x00a4: 0x0272, # CURRENCY SIGN -> LATIN SMALL LETTER N WITH LEFT HOOK
                0x00f8: 0x2205, # LATIN SMALL LETTER O WITH STROKE -> EMPTY SET
            })
        )
        data['final'] = preprocess.clean_final(
            preprocess.clean_ipa(data['final']).str.translate({
                0xf20d: 0x0264, # -> LATIN SMALL LETTER RAMS HORN
            })
        )
        data['tone'] = preprocess.clean_tone(data['tone'])

        # 删除声韵调均为空的记录
        data.replace('', pandas.NA, inplace=True)
        data.dropna(
            how='all',
            subset=['initial', 'final', 'tone'],
            inplace=True
        )

        data['did'] = did
        return data[[
            'did',
            'cid',
            'character',
            'initial',
            'final',
            'tone',
            'note'
        ]]

    @property
    def dialect_ids(self) -> list[str]:
        """
        从方言调查点信息文件获取方言 ID 列表，如果文件不存在，先从网站下载
        """

        survey = self.load_or_download('survey')
        return pandas.json_normalize(survey['dialectObj'], 'cityList')['_id'] \
            .tolist()

    @functools.cached_property
    def dialects(self) -> pandas.DataFrame:
        return self.get_dialects()

    @functools.cached_property
    def characters(self) -> pandas.DataFrame:
        """
        从方言调查标准文件获取字信息，如果文件不存在，先从网站下载
        """

        standard = self.load_or_download('standard')
        return pandas.json_normalize(standard['words']) \
            .rename(columns={'item': 'character', 'memo': 'note'}) \
            .set_index('cid')[['character', 'note']]


cache_dir = os.getenv(
    'SINCOMP_CACHE',
    os.path.join(
        os.getenv(
            'LOCALAPPDATA',
            os.path.expanduser('~')) if sys.platform.startswith('win') \
            else os.getenv(
                'XDG_DATA_HOME',
                os.path.join(os.path.expanduser('~'), '.local', 'share')
            ),
        'sincomp'
    )
)
dataset_dir = os.path.join(cache_dir, 'dataset')

ccr = CCRDataset(os.path.join(dataset_dir, 'ccr'))
mcpdict = MCPDictDataset(os.path.join(dataset_dir, 'mcpdict'))
zhongguoyuyan = ZhongguoyuyanDataset(os.path.join(dataset_dir, 'zhongguoyuyan'))
_datasets = {
    'CCR': ccr,
    'ccr': ccr,
    'xiaoxue': ccr,
    'MCPDict': mcpdict,
    'mcpdict': mcpdict,
    'zhongguoyuyan': zhongguoyuyan,
    'yubao': zhongguoyuyan
}


def list_datasets() -> list[str]:
    """
    列出所有预定义的数据集名称

    Returns:
        names: 所有预定义的数据集名称列表
    """

    return ['CCR', 'MCPDict', 'zhongguoyuyan']

def get(name: str) -> Dataset | None:
    """
    获取或加载数据集

    Parameters:
        name: 预定义数据集的名字，或本地数据集的路径

    Returns:
        dataset: 预定义或创建的数据集对象

    优先把 `name` 当作预定义数据集的名字查询，如果不成功，当作本地路径创建数据集。
    """

    try:
        return _datasets[name]

    except KeyError:
        # 不是预定义数据集，尝试把入参作为路径从本地加载
        logger.info(f'{name} is not a predefined dataset, try loading data from files.')

        if os.path.isdir(name):
            # name 是目录，使用目录下的数据创建数据集
            return FileDataset(data_dir=name)

        elif os.path.isfile(name):
            # name 是文件，直接加载数据后包装成数据集
            return Dataset(
                pandas.read_csv(name, dtype=str, encoding='utf-8'),
                name=os.path.splitext(os.path.basename(name))[0]
            )

        else:
            logger.warning(f'{name} not found!')