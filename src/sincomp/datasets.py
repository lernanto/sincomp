# -*- coding: utf-8 -*-

"""
汉语方言读音数据集

当前支持读取：
    - 汉字音典的现代方言数据，见：https://mcpdict.sourceforge.io/
    - 小学堂汉字古今音资料库的现代方言数据，见：https://xiaoxue.iis.sinica.edu.tw/ccrdata/
    - 中国语言资源保护工程采录展示平台的方言数据，见：https://zhongguoyuyan.cn/
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import functools
import io
import json
import logging
import numpy
import opencc
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
    """数据集基类"""

    def __init__(
        self,
        data: pandas.DataFrame | None = None,
        name: str | None = None
    ):
        """
        Parameters:
            data: 方言字音数据表，或另一个数据集
            name: 数据集名字
        """

        if isinstance(data, Dataset):
            data = data.data

        self._data = data
        self.name = name

    @property
    def data(self) -> pandas.DataFrame | None:
        """返回数据集内部数据"""

        return self._data

    @functools.cache
    def get_dialects(self) -> pandas.DataFrame:
        """
        返回数据集的方言点信息

        Returns:
            dialects: 方言信息表，每行为一个方言点，以方言 ID 为索引
        """

        try:
            return pandas.DataFrame(
                index=self.data['did'].drop_duplicates().dropna()
            )
        except (TypeError, KeyError):
            return pandas.DataFrame().rename_axis('did')

    @functools.cache
    def get_characters(self) -> pandas.DataFrame:
        """
        返回数据集的字信息

        Returns:
            characters: 字信息表，每行为一个字，以字 ID 为索引
        """

        if (data := self.data) is None:
            return pandas.DataFrame(columns=['character']).rename_axis('cid')

        else:
            return data.reindex(['cid', 'character'], axis=1) \
                .sort_values(['cid', 'character']) \
                .drop_duplicates() \
                .dropna(how='all')

    @property
    def dialects(self) -> pandas.DataFrame:
        return self.get_dialects()

    @property
    def characters(self) -> pandas.DataFrame:
        return self.get_characters() \
            .drop_duplicates('cid') \
            .dropna(subset='cid') \
            .set_index('cid')

    @property
    def dialect_ids(self) -> list[str]:
        """
        返回所有方言 ID

        Returns:
            ids: 方言 ID 列表
        """

        return self.get_dialects().index.to_list()

    def items(self):
        """
        依次访问每个方言点的数据
        """

        data = self.data
        for did in self.dialect_ids:
            yield did, data[data['did'] == did]

    def filter(self, dids: list[str]):
        """
        从数据集中筛选方言

        Parameters:
            dids: 保留的方言 ID 列表

        Returns:
            output: 筛选后的数据集，只包含指定的方言
        """

        data = self.data
        try:
            return Dataset(data[data['did'].isin(dids)])
        except (TypeError, KeyError):
            return Dataset()

    def __iter__(self):
        return (data for _, data in self.items())

    def __getitem__(self, key):
        data = self.data
        return None if data is None else data.__getitem__(key)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise(AttributeError(
                f'{repr(type(self).__name__)} object has no attribute {repr(name)}',
                name=name,
                obj=self
            ))

        data = self.data
        return None if data is None else data.__getattr__(name)

    def __repr__(self):
        return f'<{type(self).__name__} {repr(self.name)}>'

    def __str__(self):
        return str(self.name)


class FileDataset(Dataset):
    """
    基于文件的数据集

    数据以 CSV 形式存放在一系列文件中，每个文件是一个方言点。
    """

    def __init__(
        self,
        file_map: pandas.Series | None = None,
        path: str | None = None,
        dialect_info_path: str | None = None,
        char_info_path: str | None = None,
        name: str | None = None
    ):
        """
        Parameters:
            file_map: 方言 ID 到数据文件路径的映射表
            path: 数据集所在的目录路径
            dialect_info_path: 方言信息文件路径
            char_info_path: 字信息文件路径
        """

        if path is not None:
            path = os.path.abspath(path)

        if dialect_info_path is not None:
            dialect_info_path = os.path.abspath(dialect_info_path)
        elif path is not None and \
            os.path.isfile(p := os.path.join(path, '.dialects')):
            # 指定了数据目录且检测到方言信息文件
            dialect_info_path = p

        if char_info_path is not None:
            char_info_path = os.path.abspath(char_info_path)
        elif path is not None and \
            os.path.isfile(p := os.path.join(path, '.characters')):
            # 指定了数据目录且检测到字信息文件
            char_info_path = p

        if file_map is None and path is not None:
            # 未指定数据文件映射表但指定了数据目录，把目录下每个文件看成一个方言点，主文件名为方言 ID
            dids = []
            files = []
            for c, _, fs in os.walk(path):
                for f in fs:
                    did = os.path.splitext(f)[0]
                    p = os.path.join(c, f)
                    if p != dialect_info_path and p != char_info_path:
                        dids.append(did)
                        files.append(p)

            file_map = pandas.Series(files, index=dids)

        if name is None and path is not None:
            name = os.path.basename(path)

        super().__init__(name=name)
        self._file_map = file_map
        self._dialect_info_path = dialect_info_path
        self._char_info_path = char_info_path

    def get_data_path(self, did: str) -> str:
        """
        返回指定方言的数据文件路径

        Parameters:
            did: 指定的方言 ID

        Returns:
            path: 该方言数据文件的路径
        """

        return self._file_map.at[did]

    @classmethod
    def load_file(cls, path: str) -> pandas.DataFrame:
        """
        从文件加载单个方言点的数据

        Parameters:
            path: 方言数据文件路径

        Returns:
            data: 方言读音数据表
        """

        return pandas.read_csv(path, dtype=str, encoding='utf-8')

    def load(self, did: str) -> pandas.DataFrame:
        """
        加载指定方言点的数据

        使用独立函数的原因是方便重载加入缓存。

        Parameters:
            did: 要加载的方言 ID

        Returns:
            data: 方言读音数据表
        """

        return self.load_file(self.get_data_path(did))

    def load_dialect_info(self) -> pandas.DataFrame:
        """
        加载方言信息并返回
        """

        if self._dialect_info_path is not None:
            return pandas.read_csv(
                self._dialect_info_path,
                index_col='did',
                dtype={'did': str}
            )

    def load_char_info(self) -> pandas.DataFrame:
        """
        加载字信息并返回
        """

        if self._char_info_path is not None:
            return pandas.read_csv(self._char_info_path, dtype=str)

    @property
    def data(self) -> pandas.DataFrame:
        """
        返回所有方言读音数据

        Returns:
            output: 合并所有文件数据的长表
        """

        output = pandas.concat(self, axis=0, ignore_index=True)
        logger.debug(f'{output.shape[0]} records loaded.')
        return output

    @functools.cache
    def get_dialects(self) -> pandas.DataFrame:
        if self._dialect_info_path is None:
            return pandas.DataFrame(index=self._file_map.index)
        else:
            dialects = self.load_dialect_info()
            return dialects if self._file_map is None \
                else dialects.loc[self._file_map.index]

    @functools.cache
    def get_characters(self) -> pandas.DataFrame:
        try:
            chars = self.load_char_info()
        except FileNotFoundError:
            chars = None

        if chars is None:
            chars = super().get_characters()

        return chars

    def items(self):
        """
        依次访问每个方言点的数据
        """

        for did in self.dialect_ids:
            yield did, self.load(did)

    def iterrows(self):
        """
        依次访问每各方言点的每条记录
        """

        for data in self:
            for r in data.iterrows():
                yield r

    def filter(self, dids: list[str]):
        """
        从数据集中筛选方言

        Parameters:
            dids: 保留的方言 ID 列表

        Returns:
            output: 筛选后的数据集，只包含指定的方言
        """

        return FileDataset(
            file_map=self._file_map.loc[dids],
            dialect_info_path=self._dialect_info_path,
            char_info_path=self._char_info_path
        )

    def sample(self, *args, **kwargs) -> Dataset:
        """
        从数据集随机抽样部分方言

        Parameters:
            args, kwargs: 透传给 pandas.Serires.sample 用于抽样方言

        Returns:
            output: 包含抽样方言的数据集
        """

        return self.filter(self.dialects.sample(*args, **kwargs).index)

    def shuffle(
        self,
        random_state: numpy.random.RandomState | int | None = None
    ) -> Dataset:
        """
        随机打乱数据集中方言的顺序

        Parameters:
            random_state: 用于控制打乱结果

        Returns:
            output: 内容相同的数据集，但方言的顺序随机打乱了
        """

        dids = numpy.asarray(self.dialect_ids)
        numpy.random.RandomState(random_state).shuffle(dids)
        return self.filter(dids)

    def append(self, other: Dataset) -> Dataset:
        """
        把另一个数据集追加到本数据集后面

        Parameters:
            other: 另一个数据集，必须也是 FileDataset

        Returns:
            output: 合并了两个数据集文件的新数据集
        """

        return FileDataset(
            file_map=pandas.concat([self._file_map, other._file_map])
        )

    def __len__(self) -> int:
        """
        返回数据集包含的方言数
        """

        return len(self.dialect_ids)

    def __add__(self, other: Dataset) -> Dataset:
        return self.append(other)


class FileCacheDataset(FileDataset):
    """
    使用本地文件作为缓存的数据集

    首次加载一个方言数据时，会加载原始数据并处理成 FileDataset 能处理的格式，保存在本地缓存文件。
    以后加载会加载缓存文件，以加快加载速度。
    """

    def __init__(
        self,
        cache_dir: str,
        name: str | None = None,
        dialect_ids: list[str] | None = None
    ):
        """
        Parameters:
            cache_dir: 缓存文件所在目录路径
            dialect_ids: 数据集包含的所有方言 ID 列表
        """

        cache_dir = os.path.abspath(cache_dir)
        super().__init__(
            dialect_info_path=os.path.join(cache_dir, '.dialects'),
            char_info_path=os.path.join(cache_dir, '.characters'),
            name=name
        )
        self._cache_dir = cache_dir
        self._dialect_ids = dialect_ids

    @functools.cache
    def get_dialects(self) -> pandas.DataFrame:
        try:
            dialects = super().load_dialect_info()
        except FileNotFoundError:
            dialects = self.load_dialect_info()
            os.makedirs(self._cache_dir, exist_ok=True)
            dialects.to_csv(
                self._dialect_info_path,
                encoding='utf-8',
                lineterminator='\n'
            )

        return dialects if self._dialect_ids is None \
            else dialects.loc[self._dialect_ids]

    @functools.cache
    def get_characters(self) -> pandas.DataFrame:
        try:
            return super().load_char_info()
        except FileNotFoundError:
            chars = super().get_characters()
            os.makedirs(self._cache_dir, exist_ok=True)
            chars.to_csv(
                self._char_info_path,
                index=False,
                encoding='utf-8',
                lineterminator='\n'
            )
            return chars

    def get_data_path(self, did: str) -> str:
        """
        返回指定方言的数据文件路径

        Parameters:
            did: 指定的方言 ID

        Returns:
            path: 该方言数据文件的路径
        """

        return os.path.join(self._cache_dir, did)

    def load(self, did: str) -> pandas.DataFrame:
        """
        从缓存或原始数据加载一个方言的数据

        Parameter:
            did: 要加载的方言 ID

        Returns:
            data: 加载的方言数据表

        如果已存在缓存文件，直接从缓存文件读取数据返回，否则从原始数据加载并写入缓存文件。
        由于原始数据不一定按单个方言存储，因此可能创建多个缓存文件。
        """

        try:
            # 如果已存在缓存文件，直接读取
            return super().load_file(self.get_data_path(did))

        except FileNotFoundError:
            # 不存在缓存文件，从原始数据读取
            # 因为原始数据可能一次不止加载一个方言，因此返回的是方言 ID 和数据表的列表
            data_list = self.load_data(did)
            # 写入文件缓存
            for i, d in data_list:
                if not os.path.isfile(path := self.get_data_path(i)):
                    logger.info(f'create cache file {path}.')
                    os.makedirs(self._cache_dir, exist_ok=True)
                    d.to_csv(
                        path,
                        index=False,
                        encoding='utf-8',
                        lineterminator='\n'
                    )

                if i == did:
                    data = d

            return data

    def clear_cache(self):
        """
        删除所有缓存文件
        """

        for path in self._dialect_info_path, self._char_info_path:
            logger.info(f'remove cache file {path}.')
            try:
                os.remove(path)
            except FileNotFoundError:
                ...
            except OSError as e:
                logger.warning(e)
           
        for did in self.dialect_ids:
            path = self.get_data_path(did)
            logger.info(f'remove cache file {path}.')
            try:
                os.remove(path)
            except FileNotFoundError:
                ...
            except OSError as e:
                logger.warning(e)

        # 删除缓存目录
        try:
            logger.info(f'remove cache directory {self._cache_dir}.')
            os.rmdir(self._cache_dir)
        except OSError as e:
            logger.warning(e)

    def refresh(self):
        """
        强制更新全部缓存文件
        """

        self.clear_cache()
        self.get_dialects.cache_clear()
        self.get_dialects()
        self.get_characters.cache_clear()
        self.get_characters()
        for did in self.dialect_ids:
            self.load(did)


class MCPDictDataset(FileCacheDataset):
    """
    汉字音典方言数据集

    见：https://mcpdict.sourceforge.io/。
    """

    def __init__(
        self,
        cache_dir: str,
        superscript_tone: bool = False,
        na: str | None = None,
        empty: str | None = '∅',
        name: str = 'MCPDict',
        **kwargs
    ):
        """
        Parameters:
            cache_dir: 缓存文件所在目录路径
            superscript_tone: 为真时，把声调中的普通数字转成上标数字
            na: 代表缺失的字符串，为 None 时保持原状
            empty: 代表零声母/零韵母/零声调的字符串，为 None 时保持原状
            name: 数据集名称
            kwargs: 透传给 `FileCacheDataset`
        """

        cache_dir = os.path.abspath(cache_dir)
        super().__init__(cache_dir, name=name, **kwargs)
        self._path = os.path.join(cache_dir, 'tools', 'tables', 'output')
        self._superscript_tone = superscript_tone
        self._na = na
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

        logger.info(f'downloading {url}...')

        with urllib.request.urlopen(url) as res:
            with zipfile.ZipFile(io.BytesIO(res.read())) as zf:
                os.makedirs(
                    os.path.join(output, 'tools', 'tables', 'output'),
                    exist_ok=True
                )
                logger.info(f'extracting files to {output}...')

                for info in zf.infolist():
                    # 路径第一段是带版本号的项目名，需去除
                    path = info.filename.partition('/')[2]
                    # 把字音数据目录的所有文件解压到目标路径
                    if not info.is_dir() and path.startswith('tools/tables/output/'):
                        logger.info(f'extracting {info.filename}...')
                        path = os.path.join(*[output] + path.split('/'))
                        with open(path, 'wb') as of:
                            of.write(zf.read(info))

        logger.info('done.')

    @functools.cache
    def load_dialect_info_raw(self) -> pandas.DataFrame:
        """
        加载方言点信息

        Returns:
            info: 方言信息表
        """

        if not os.path.isdir(self._path):
            # 数据文件不存在，先从汉字音典项目页面下载数据
            logger.info(
                'run for the first time, download data from Web, '
                'this may take a while.'
            )
            self.download(self._cache_dir)

        info = pandas.read_json(
            os.path.join(self._path, '_詳情.json'),
            orient='index',
            encoding='utf-8'
        )

        # 汉典的方言数据实际来自小学堂，已收录在小学堂数据集，此处剔除
        # 汉字音典数据包含历史拟音、域外方音和一些拼音方案，只取用国际音标注音的现代方言数据
        info = info[
            (info['文件格式'] != '漢典') \
            & (~info['地圖集二分區'].isin(['歷史音', '民族語', '域外方音', '戲劇'])) \
            & (~info.index.str.match('^1[0-9]{3}') | info.index.isin([
                '1935永明',
                '1935南昌',
                '1935醴陵',
                '1935長沙',
            ])) \
            & (~info.index.isin([
                '普通話',
                '鄕音字類',
                '淸末寧波',
                '淸末溫州',
                '訓詁諧音',
                '湘音檢字',
                '香港',
                '臺灣'
            ])) \
            & ((self._path + os.sep + info.index + '.tsv').map(os.path.isfile))
        ]

        if info.shape[0] == 0:
            logger.warning(f'no valid data in {self._path}.')
            return

        # 使用简称作为方言 ID
        return info.set_index('簡稱', drop=False)

    def load_dialect_info(self) -> pandas.DataFrame:
        """
        加载方言点信息

        Returns:
            info: 规范化的方言信息表
        """

        info = self.load_dialect_info_raw()

        # 解析方言分类
        cat = info['地圖集二分區'].str.split('-')
        # 乡话使用了异体字，OpenCC 无法转成简体，特殊处理
        info = info.assign(
            group=cat.str[0].replace('鄕話', '鄉話'),
            cluster=cat.str[1],
            subcluster=cat.str[2]
        )

        mask = info['group'].str.endswith('官話') \
            | info['group'].str.endswith('官话')
        info.loc[mask, 'subgroup'] = info.loc[mask, 'group']
        info.loc[mask, 'group'] = '官話'

        # 原始分区不分平话和土话，根据子分区信息尽量分开
        mask = info['group'] == '平話和土話'
        info.loc[mask, 'group'] = numpy.where(
            info.loc[mask, 'cluster'].isin(['桂南片', '桂北片']),
            '平話',
            '土話'
        )

        # 解析经纬度
        info[['latitude', 'longitude']] = info['經緯度'].str.partition(',') \
            .iloc[:, [2, 0]].astype(float, errors='ignore')

        info = info.rename_axis('did').rename(columns={
            '語言': 'name',
            '簡稱': 'spot',
            '省': 'province',
            '市': 'city',
            '縣': 'county',
            '鎮': 'town',
            '村': 'village',
        })

        # 把方言信息转换成简体中文
        info.update(info[[
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

        return info.reindex([
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

    @property
    def tone_map(self) -> dict[str, tuple[dict[str, str], dict[str, str]]]:
        """
        从方言详情提取声调调值和调类的映射表

        Returns:
            tone_map: 方言 ID 到声调映射表的映射表，其值又是映射表的二元组，
                前者为调号到调值的映射表，后者为调号到调类的映射表
        """

        tm = {}
        for i, m in self.load_dialect_info_raw()['聲調'].map(json.loads).items():
            tone = {}
            cat = {}
            for k, v in m.items():
                # 少数连读声调有特殊符号，暂时去除
                tone[k] = re.sub(f'[^{"".join(preprocess._TONES)}]', '', v[0])
                cat[k] = v[3]
            tm[i] = tone, cat

        return tm

    @classmethod
    def load_raw(cls, did: str, path: str) -> pandas.DataFrame:
        """
        从原始文件加载指定方言读音数据

        Parameters:
            did: 要加载的方言 ID
            path: 要加载的数据文件路径

        Returns:
            data: 加载的读音数据表
        """

        data = pandas.read_csv(
            path,
            sep='\t',
            header=None,
            names=['漢字', '音標', '解釋'],
            dtype=str,
            na_values={'漢字': '\u25a1'},   # 方框代表有音无字
            comment='#',
            encoding='utf-8'
        )

        data['音標'] = data['音標'].str.translate({
            0x008f: 0x027f, # -> LATIN SMALL LETTER REVERSED R WITH FISHHOOK
            0x0090: 0x0285, # -> LATIN SMALL LETTER SQUAT REVERSED ESH
        })

        return data.replace('', pandas.NA)

    def load_data(self, did: str) -> list[tuple[str, pandas.DataFrame]]:
        """
        加载方言读音数据并清理

        Parameters:
            did: 要加载的方言 ID

        Returns:
            did: 同参数 `did`
            data: 方言读音数据表
        """

        path = os.path.join(self._path, did + '.tsv')
        logger.info(f'load data from {path}.')
        data = self.load_raw(did, path).assign(did=did)

        # 把原始读音切分成声母、韵母、声调
        seg = data.pop('音標').str.extract(r'([^0-9]*)([0-9][0-9a-z]*)?')
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

        # 替换列名为统一的名称
        data.rename(columns={'漢字': 'character', '解釋': 'note'}, inplace=True)

        if self._superscript_tone:
            # 把声调中的普通数字转成上标数字
            data['tone'] = preprocess.tone2super(data['tone'])

        # 根据需要替换缺失值及空值
        if self._na is not None:
            data.update(data[
                ['initial', 'final', 'tone', 'tone_category']
            ].fillna(self._na))

        if self._empty is not None:
            data.replace(
                {'initial': '', 'final': '', 'tone': ''},
                self._empty,
                inplace=True
            )

        return ((did, data.reindex([
            'did',
            'cid',
            'character',
            'initial',
            'final',
            'tone',
            'tone_category',
            'note'
        ], axis=1)),)

    def filter(self, dids: list[str]):
        """
        从数据集中筛选方言

        Parameters:
            dids: 保留的方言 ID 列表

        Returns:
            output: 筛选后的数据集，只包含指定的方言

        TODO: 当前对 `filter` 返回的数据集调用 `get_characters` 会把过滤后的字表写入缓存文件，
        而对原始数据集调用 `get_characters` 也会读取该缓存文件，导致显示的字表和实际不一致。
        """

        return MCPDictDataset(
            self._cache_dir,
            self._superscript_tone,
            self._na,
            self._empty,
            None,
            dialect_ids=list(dids)
        )


class CCRDataset(FileCacheDataset):
    """
    小学堂汉字古今音资料库的现代方言数据集

    见：https://xiaoxue.iis.sinica.edu.tw/ccrdata/。
    """

    def __init__(
        self,
        cache_dir: str,
        superscript_tone: bool = False,
        na: str | None = None,
        empty: str | None = None,
        name: str = 'CCR',
        **kwargs
    ):
        """
        Parameters:
            cache_dir: 缓存文件所在目录路径
            superscript_tone: 为真时，把声调中的普通数字转成上标数字
            na: 代表缺失的字符串，为 None 时保持原状
            empty: 代表零声母/零韵母/零声调的字符串，为 None 时保持原状
            name: 数据集名称
            kwargs: 透传给 `FileCacheDataset`
        """

        cache_dir = os.path.abspath(cache_dir)
        super().__init__(cache_dir, name=name, **kwargs)
        self._path = cache_dir
        self._superscript_tone = superscript_tone
        self._na = na
        self._empty = empty

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

    @functools.cache
    def load_dialect_info_raw(self) -> pandas.DataFrame:
        """
        加载方言点信息

        Returns:
            info: 方言点信息数据表，只包含文件中编号非空的方言点
        """

        info = pandas.read_csv(
            os.path.join(os.path.dirname(__file__), 'ccr_dialect_info.csv'),
            dtype={'編號': str}
        ).dropna(subset=['編號'])

        # 各方言数据下载地址及下载后解压的文件路径
        return info.assign(
            url='https://xiaoxue.iis.sinica.edu.tw/ccrdata/file/' \
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
            }),
            path=self._path + os.sep + info['編號'] + ' ' \
                + info['方言'].where(info['方言'] != '官話', info['區']) + \
                '_' + info['方言點'] + '.xlsx'
        ).set_index('編號')

    def load_dialect_info(self) -> pandas.DataFrame:
        """
        加载方言点信息

        Returns:
            info: 规范化的方言点信息数据表
        """

        info = self.load_dialect_info_raw()

        # 少数方言名称转成更通行的名称；部分方言点包含来源文献，删除
        info = info.replace({'方言': {'客語': '客家話', '其他土話': '土話'}}) \
            .assign(區=self.clean_subgroup(info['區'])) \
            .assign(name=info['方言點'].str.replace(
                r'\(安徽省志\)|\(珠江三角洲\)|\(客贛方言調查報告\)|\(廣西漢語方言\)'
                r'|\(平話音韻研究\)|\(廣東閩方言\)|\(漢語方音字匯\)|\(當代吳語\)',
                '',
                regex=True
            )) \
            .rename_axis('did') \
            .rename(columns={
                '方言': 'group',
                '區': 'subgroup',
                '片／小區': 'cluster',
                '小片': 'subcluster',
                '方言點': 'spot',
                '緯度': 'latitude',
                '經度': 'longitude'
            })

        # 把方言信息转换成简体中文
        info.update(
            info[['group', 'subgroup', 'cluster', 'subcluster', 'spot']] \
                .map(opencc.OpenCC('t2s').convert, na_action='ignore')
        )

        return info.reindex([
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

    def load_char_info(self) -> pandas.DataFrame:
        """
        加载字信息

        Returns:
            info: 字信息数据表
        """

        return pandas.read_csv(
            os.path.join(os.path.dirname(__file__), 'ccr_char_info.csv'),
            dtype=str
        ) \
            .rename(columns={'字號': 'cid', '字': 'character'})[['cid', 'character']]

    @staticmethod
    @retry.retry(exceptions=urllib.error.URLError, tries=3, delay=1)
    def download(url: str, output: str) -> None:
        """
        从小学堂网站下载方言读音数据

        Parameters:
            url: 下载地址
            output: 保存下载解压文件的本地目录
        """

        logger.info(f'downloading {url}...')

        # 设置 User-Agent，否则请求会被拒绝
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as res:
            with zipfile.ZipFile(io.BytesIO(res.read())) as zf:
                logger.info(f'extracting files to {output}...')
                os.makedirs(output, exist_ok=True)

                for info in zf.infolist():
                    # 压缩包路径编码为 Big5，但 zipfile 默认用 CP437 解码，需重新用 Big5 解码
                    try:
                        fname = info.filename.encode('cp437').decode('big5')
                    except UnicodeError:
                        fname = info.filename
                    # 改正文件名中的别字
                    fname = fname.replace('閔', '閩')

                    logger.info(f'extracting {fname}...')
                    with open(os.path.join(output, fname), 'wb') as of:
                        of.write(zf.read(info))

        logger.info('done.')

    @classmethod
    def load_raw(cls, id: str, path: str) -> pandas.DataFrame:
        """
        从 xlsx 文件加载方言读音数据

        Parameters:
            id: 要加载的方言的原始 ID（未加前缀）
            path: 要加载的数据文件路径

        Returns:
            data: 加载的读音数据表
        """

        data = pandas.read_excel(path, dtype=str)

        # 少数文件列的命名和其他文件不一致，统一成最常用的
        data.rename(columns={
            'Order': '字號',
            'Char': '字',
            'ShengMu': '聲母',
            'YunMu': '韻母',
            'DiaoZhi': '調值',
            'DiaoLei': '調類',
            'Comment': '備註'
        }, inplace=True)

        # 多音字每个读音为一行，但有些多音字声韵调部分相同的，只有其中一行标了数据，
        # 其他行为空。对于这些空缺，使用同字第一个非空的的读音填充
        data.fillna(
            data.groupby('字號')[['聲母', '韻母', '調值', '調類']].transform('first'),
            inplace=True
        )

        # 清洗数据集特有的错误
        if id == '118':
            data['韻母'] = data['韻母'].str.translate({
                0x003f: 0x028f, # QUESTION MARK -> LATIN LETTER SMALL CAPITAL Y
            })
        elif id == '178':
            data['聲母'] = data['聲母'].str.translate({
                0x0237: 0x0255, # LATIN SMALL LETTER DOTLESS J -> LATIN SMALL LETTER C WITH CURL
            })

        # 清洗读音 IPA
        data['聲母'] = preprocess.clean_ipa(data['聲母'])
        data['韻母'] = preprocess.clean_ipa(data['韻母'])
        data['調值'] = data['調值'].str.translate({
            0x0030: 0x2205, # DIGIT ZERO -> EMPTY SET
        })

        return data.replace('', pandas.NA)

    def load_data(self, did: str) -> list[tuple[str, pandas.DataFrame]]:
        """
        加载方言字音数据

        Parameters:
            did: 要加载的方言 ID

        Returns:
            did: 同参数 `did`
            data: 方言读音数据表

        如果要加载的数据文件不存在，先从网站下载。
        """

        path = self.load_dialect_info_raw().at[did, 'path']
        if not os.path.isfile(path):
            # 方言数据文件不存在，从网站下载
            self.download(
                self.load_dialect_info_raw().at[did, 'url'],
                self._cache_dir
            )

        logger.info(f'loading data from {path}...')
        data = self.load_raw(did, path).assign(did=did)

        # 清洗读音数据。一个格子可能记录了多个音，用点分隔，只取第一个
        data['聲母'] = preprocess.clean_initial(data['聲母'].str.split('.').str[0])
        data['韻母'] = preprocess.clean_final(data['韻母'].str.split('.').str[0])
        data['調值'] = preprocess.clean_tone(data['調值'].str.split('.').str[0])
        data['調類'] = data['調類'].str.split('.').str[0] \
            .str.replace(r'[^上中下變陰陽平去入輕聲]', '', regex=True)

        data.replace(
            {'聲母': '', '韻母': '', '調值': '', '調類': ''},
            pandas.NA,
            inplace=True
        )
        # 删除声韵调均为空的记录
        data.dropna(how='all', subset=['聲母', '韻母', '調值'], inplace=True)

        if self._superscript_tone:
            # 把声调中的普通数字转成上标数字
            data['調值'] = preprocess.tone2super(data['調值'])

        # 根据需要替换缺失值及空值
        if self._na is not None:
            data[['聲母', '韻母', '調值', '調類']] \
                = data[['聲母', '韻母', '調值', '調類']].fillna(self._na)

        if self._empty is not None:
            data.replace(
                {'聲母': '∅', '韻母': '∅', '調值': '∅'},
                self._empty,
                inplace=True
            )

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

        return ((did, data[[
            'did',
            'cid',
            'character',
            'initial',
            'final',
            'tone',
            'tone_category',
            'note'
        ]]),)

    def filter(self, dids: list[str]):
        """
        从数据集中筛选方言

        Parameters:
            dids: 保留的方言 ID 列表

        Returns:
            output: 筛选后的数据集，只包含指定的方言
        """

        return CCRDataset(
            self._cache_dir,
            self._superscript_tone,
            self._na,
            self._empty,
            None,
            dialect_ids=list(dids)
        )


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


class ZhongguoyuyanDataset(FileCacheDataset):
    """
    中国语言资源保护工程采录展示平台的方言数据集

    见：https://zhongguoyuyan.cn/。
    """


    def __init__(
        self,
        cache_dir: str,
        superscript_tone: bool = False,
        na: str | None = None,
        empty: str | None = None,
        name: str = 'zhongguoyuyan',
        downloader_kwargs: dict = {},
        **kwargs
    ):
        """
        Parameters:
            cache_dir: 缓存文件所在目录路径
            superscript_tone: 为真时，把声调中的普通数字转成上标数字
            na: 代表缺失数据的字符串，为 None 时保持原状
            empty: 代表零声母/零韵母/零声调的字符串，为 None 时保持原状
            name: 数据集名称
            downloader_kwargs: 传给下载器的参数
            kwargs: 透传给 `FileCacheDataset`
        """

        super().__init__(cache_dir, name=name, **kwargs)
        self._superscript_tone = superscript_tone
        self._na = na
        self._empty = empty
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

    def load_dialect_info(self) -> pandas.DataFrame:
        """
        读取方言点信息

        Returns:
            info: 方言点信息数据表
        """

        survey = self.load_or_download('survey')
        info = pandas.json_normalize(survey['dialectObj'], 'cityList')
        info.set_index('_id', inplace=True)
        info['city'] = info['city'].where(
            ~info['city'].str.match(r'^[(（]无[)）]$', na=True)
        )
        info['name'] = info['filepath'].str.extract(r'^[^/]+/([^#]+)#?需交文件电子版')

        return info.rename_axis('did').reindex([
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

    def load_char_info(self) -> pandas.DataFrame:
        """
        读取字信息

        Returns:
            info: 字信息数据表
        """

        standard = self.load_or_download('standard')
        return pandas.json_normalize(standard['words']) \
            .rename(columns={'item': 'character', 'memo': 'note'}) \
            [['cid', 'character', 'note']]

    def load_raw(self, id: str) -> pandas.DataFrame:
        """
        加载指定方言读音数据

        Parameters:
            id: 要加载的方言的原始 ID（未加前缀）

        Returns:
            data: 加载的读音数据表

        语保数据文件包含了单字、词汇和语法，其中单字包含了老年男性和青年男性两个发音人的数据，
        当前只取其中单字的老年男性数据。如果文件不存在，先从网站下载。
        """

        point = self.load_or_download(id)
        data = pandas.json_normalize(
            point['data']['resourceList'][0]['items'], 
            'records',
            ['iid', 'name']
        )

        # 清洗数据集特有的错误
        if id == '02135':
            data['finals'] = data['finals'].str.translate({
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
        data['initial'] = preprocess.clean_ipa(data['initial']).str.translate({
            0x00a4: 0x0272, # CURRENCY SIGN -> LATIN SMALL LETTER N WITH LEFT HOOK
            0x00f8: 0x2205, # LATIN SMALL LETTER O WITH STROKE -> EMPTY SET
        })
        data['finals'] = preprocess.clean_ipa(data['finals']).str.translate({
            0xf20d: 0x0264, # -> LATIN SMALL LETTER RAMS HORN
        })

        return data.replace('', pandas.NA)

    def load_data(self, did: str) -> list[tuple[str, pandas.DataFrame]]:
        """
        加载方言字音数据

        Parameters:
            ids: 要加载的方言列表，当为空时，加载路径中所有方言数据

        Returns:
            data: 方言字音表
        """

        data = self.load_raw(did).assign(did=did)

        # 清洗读音数据
        data['initial'] = preprocess.clean_initial(data['initial'])
        data['finals'] = preprocess.clean_final(data['finals'])
        data['tone'] = preprocess.clean_tone(data['tone'])

        data.replace(
            {'initial': '', 'finals': '', 'tone': ''},
            pandas.NA,
            inplace=True
        )
        # 删除声韵调均为空的记录
        data.dropna(
            how='all',
            subset=['initial', 'finals', 'tone'],
            inplace=True
        )

        if self._superscript_tone:
            # 把声调中的普通数字转成上标数字
            data['tone'] = preprocess.tone2super(data['tone'])

        # 根据需要替换缺失值及空值
        if self._na is not None:
            data[['initial', 'finals', 'tone']] \
                = data[['initial', 'finals', 'tone']].fillna(self._na)

        if self._empty is not None:
            data.replace(
                {'initial': '∅', 'finals': '∅', 'tone': '∅'},
                self._empty,
                inplace=True
            )

        # 替换列名为统一的名称
        data.rename(columns={
            'iid': 'cid',
            'name': 'character',
            'finals': 'final',
            'memo': 'note'
        }, inplace=True)

        return ((did, data[[
            'did',
            'cid',
            'character',
            'initial',
            'final',
            'tone',
            'note'
        ]]),)

    def filter(self, dids: list[str]):
        """
        从数据集中筛选方言

        Parameters:
            dids: 保留的方言 ID 列表

        Returns:
            output: 筛选后的数据集，只包含指定的方言
        """

        return ZhongguoyuyanDataset(
            self._cache_dir,
            self._superscript_tone,
            self._na,
            self._empty,
            None,
            dialect_ids=list(dids)
        )


cache_dir = os.environ.get(
    'SINCOMP_CACHE',
    os.path.join(
        os.environ.get(
            'LOCALAPPDATA',
            os.path.expanduser('~')) if sys.platform.startswith('win') \
            else os.environ.get(
                'XDG_DATA_HOME',
                os.path.join(os.path.expanduser('~'), '.local', 'share')
            ),
        'sincomp',
        'datasets'
    )
)

ccr = CCRDataset(os.path.join(cache_dir, 'ccr'))
mcpdict = MCPDictDataset(os.path.join(cache_dir, 'mcpdict'))
zhongguoyuyan = ZhongguoyuyanDataset(os.path.join(cache_dir, 'zhongguoyuyan'))
_datasets = {
    'CCR': ccr,
    'ccr': ccr,
    'xiaoxue': ccr,
    'MCPDict': mcpdict,
    'mcpdict': mcpdict,
    'zhongguoyuyan': zhongguoyuyan,
    'yubao': zhongguoyuyan
}


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
            return FileDataset(path=name)

        elif os.path.isfile(name):
            # name 是文件，直接加载数据后包装成数据集
            return Dataset(
                pandas.read_csv(name, dtype=str, encoding='utf-8'),
                name=os.path.splitext(os.path.basename(name))[0]
            )

        else:
            logger.warning(f'{name} not found!')


if __name__ == '__main__':
    # 刷新所有数据集的缓存文件
    print('refresh cache files for all datasets. this may take a while.')
    for name in 'CCR', 'MCPDict', 'zhongguoyuyan':
        dataset = get(name)
        if dataset is not None:
            print(f'refreshing {dataset.name}...')
            try:
                dataset.refresh()
                print(f'done writing {dataset.dialects.shape[0]} dialects.')
            except:
                logging.warning(f'cannot refresh {dataset.name}!', exc_info=True)