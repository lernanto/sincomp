# -*- coding: utf-8 -*-

"""
汉语方言读音数据集

当前支持读取：
    - 汉字音典的现代方言数据，见：https://mcpdict.sourceforge.io/
    - 小学堂汉字古今音资料库的现代方言数据，见：https://xiaoxue.iis.sinica.edu.tw/ccr
    - 中国语言资源保护工程采录展示平台的方言数据，见：https://zhongguoyuyan.cn/
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import os
import logging
import collections
import pandas
import numpy
import json
import functools
import opencc
from sklearn.neighbors import KNeighborsClassifier

from . import preprocess


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
        name: str,
        data: pandas.DataFrame | None = None,
        metadata: dict = dict()
    ):
        """
        Parameters:
            name: 数据集名称
            data: 方言字音数据表
            metadata: 数据集元信息
        """

        self.name = name
        self._data = data
        self.metadata = metadata

    @property
    def data(self) -> pandas.DataFrame | None:
        return self._data

    @functools.lru_cache
    def _force_complete(
        self,
        columns: list[str] | None = None
    ) -> pandas.DataFrame:
        """
        保证一行数据指定的列都有值，否则删除该读音

        Parameters:
            columns: 需要保证有值的列，空表示所有列

        Returns:
            output: 删除不完整读音后的方言字音数据表
        """

        if columns is None:
            columns = self.data.columns
        else:
            columns = list(columns)

        invalid = (self.data[columns].isna() | (self.data[columns] == '')) \
            .any(axis=1)
        return self.data.drop(self.data.index[invalid])

    def force_complete(
        self,
        columns: list[str] | None = ['initial', 'final', 'tone']
    ):
        """
        保证一行数据指定的列都有值，否则删除该读音

        Parameters:
            columns: 需要保证有值的列

        Returns:
            other (`sinetym.dataset.Dataset`): 删除不完整读音后的方言字音数据集
        """

        if self.data is None:
            return Dataset(self.name, metadata=self.metadata)

        try:
            columns = tuple(columns)
        except ValueError:
            ...

        return Dataset(
            self.name,
            data=self._force_complete(columns=columns),
            metadata=self.metadata
        )

    @functools.lru_cache
    def _transform(
        self,
        index: str = 'did',
        values: list[str] | None = None,
        aggfunc: str | collections.abc.Callable = ' '.join
    ) -> pandas.DataFrame:
        """
        把方言读音数据长表转换为宽表

        当 index 为 did 时，以地点为行，字为列，声韵调为子列。
        当 index 为 cid 时，以字为行，地点为列，声韵调为子列。

        Parameters:
            index: 指明以原始表的哪一列为行，did 一个地点为一行，cid 一个字为一行
            values: 用于变换的列，变换后成为二级列，为空保留所有列
            aggfunc: 相同的 did 和 cid 有多个记录的，使用 aggfunc 函数合并

        Returns:
            output: 变换格式后的数据表
        """

        output = self.data.pivot_table(
            values,
            index=index,
            columns='cid' if index == 'did' else 'did',
            aggfunc=aggfunc,
            fill_value='',
            sort=False
        )

        # 如果列名为多层级，把指定的列名上移到最高层级
        if output.columns.nlevels > 1:
            output = output.swaplevel(axis=1).reindex(
                pandas.MultiIndex.from_product((
                    output.columns.levels[1],
                    output.columns.levels[0]
                )),
                axis=1
            )

        return output

    def transform(
        self,
        index: str = 'did',
        values: list[str] = ['initial', 'final', 'tone'],
        aggfunc: collections.abc.Callable | str = ' '.join
    ):
        """
        把方言读音数据长表转换为宽表

        当 index 为 did 时，以地点为行，字为列，声韵调为子列。
        当 index 为 cid 时，以字为行，地点为列，声韵调为子列。

        Parameters:
            index: 指明以原始表的哪一列为行，did 一个地点为一行，cid 一个字为一行
            values: 用于变换的列，变换后成为二级列
            aggfunc: 相同的 did 和 cid 有多个记录的，使用 aggfunc 函数合并

        Returns:
            other: 变换格式后的数据集
        """

        if self.data is None:
            return Dataset(self.name, metadata=self.metadata)

        try:
            values = tuple(values)
        except (TypeError, ValueError):
            ...

        return Dataset(
            self.name,
            data=self._transform(index, values, aggfunc),
            metadata=self.metadata
        )

    def __iter__(self):
        return iter(()) if self.data is None else iter(self.data)

    def __getitem__(self, key):
        return None if self.data is None else self.data.__getitem__(key)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise(AttributeError(
                f'{repr(type(self).__name__)} object has no attribute {repr(name)}',
                name=name,
                obj=self
            ))

        return None if self.data is None else self.data.__getattr__(name)

    def __repr__(self):
        return f'{type(self).__name__}({repr(self.name)})'

    def __str__(self):
        return self.name

def concat(*datasets: list[Dataset | pandas.DataFrame]) -> Dataset:
    """
    把多个数据集按顺序拼接为一个

    Parameters:
        datasets: 待拼接的数据集

    Returns:
        output: 拼接后的数据集
    """

    return Dataset(
        'dataset',
        data=pandas.concat(
            [d.data if isinstance(d, Dataset) else d for d in datasets],
            axis=0,
            ignore_index=True
        )
    )


class MCPDictDataset(Dataset):
    """
    汉字音典方言数据集

    见：https://mcpdict.sourceforge.io/。
    """

    def __init__(
        self,
        path: str,
        uniform_name: bool = True,
        uniform_info: bool = True,
        did_prefix: str | None = 'M',
        cid_prefix: str | None = 'M',
        normalize: bool = True,
        superscript_tone: bool = False,
        na: str | None = None,
        empty: str | None = '∅'
    ):
        """
        Parameters:
            path: 数据集所在的基础路径
            uniform_name: 为真时，把数据列名转为通用名称
            uniform_info: 小学堂原始数据为繁体中文，把方言信息转换成简体中文
            did_prefix: 非空时在方言 ID 添加该前缀
            cid_prefix: 非空时在字 ID 添加该前缀
            normalize: 为真时，进一步清洗读音数据，改写为规范的形式
            superscript_tone: 为真时，把声调中的普通数字转成上标数字
            na: 代表缺失的字符串，为 None 时保持原状
            empty: 代表零声母/零韵母/零声调的字符串，为 None 时保持原状
        """

        super().__init__('mcpdict')

        self._path = os.path.join(path, 'tools', 'tables', 'output')
        self._uniform_name = uniform_name
        self._uniform_info = uniform_info
        self._did_prefix = did_prefix
        self._cid_prefix = cid_prefix
        self._normalize = normalize
        self._superscript_tone = superscript_tone
        self._na = na
        self._empty = empty

        info = self.load_dialect_info()

        # 从方言详情提取声调调值和调类的映射表
        self._tone_map = {}
        for i, m in info.pop('聲調').map(json.loads).items():
            tone = {}
            cat = {}
            for k, v in m.items():
                tone[k] = v[0]
                cat[k] = v[3]
            self._tone_map[i] = tone, cat

        self._dialect_info = info
        self.metadata['dialect_info'] = info

    def load_dialect_info(self) -> pandas.DataFrame:
        """
        加载方言点信息

        Returns:
            info: 方言信息表
        """

        info = pandas.read_json(
            os.path.join(self._path, '_詳情.json'),
            orient='index',
            encoding='utf-8'
        )

        info['path'] = self._path + os.sep + info.index + '.tsv'
        # 汉典的方言数据实际来自小学堂，已收录在小学堂数据集，此处剔除
        # 汉字音典数据包含历史拟音、域外方音和一些拼音方案，只取用国际音标注音的现代方言数据
        info = info[
            (info['文件格式'] != '漢典') \
            & (~info['音典分區'].isin(['歷史音,', '域外方音,', '戲劇,'])) \
            & (~info.index.isin([
                '普通話',
                '1900梅惠',
                '1926綜合',
                '鶴山沙坪',
                '1884甯城',
                '1890會城',
                '香港',
                '臺灣',
                '劍川金華白語',
                '武鳴壯語',
                '臨高話'
            ])) \
            & (info['path'].map(os.path.isfile))
        ]

        if info.shape[0] == 0:
            logging.warning(f'no valid data in {self._path} .')
            return

        # 使用音典排序作为方言 ID
        info.set_index('音典排序', inplace=True)
        if self._did_prefix is not None:
            info.set_index(self._did_prefix + info.index, inplace=True)

        # 解析方言分类
        cat = info.pop('地圖集二分區').str.partition(',').iloc[:, 0].str.split('-')
        info['group'] = cat.str[0]
        info['cluster'] = cat.str[1]
        info['subcluster'] = cat.str[2]
        mask = info['group'].str.endswith('官話')
        info.loc[mask, 'subgroup'] = info.loc[mask, 'group']
        info.loc[mask, 'group'] = '官話'

        # 解析经纬度
        info[['latitude', 'longitude']] = info.pop('經緯度').str.partition(',') \
            .iloc[:, [2, 0]].astype(float, errors='ignore')

        if self._uniform_info:
            # 把方言信息转换成简体中文
            info.update(info[[
                '簡稱',
                '省',
                '市',
                '縣',
                '鎮',
                '村',
                'group',
                'subgroup',
                'cluster',
                'subcluster'
            ]].map(opencc.OpenCC('t2s').convert, na_action='ignore'))

        if self._uniform_name:
            info.index.rename('did', inplace=True)
            info.rename(columns={
                '簡稱': 'spot',
                '省': 'province',
                '市': 'city',
                '縣': 'county',
                '鎮': 'town',
                '村': 'village',
            }, inplace=True)

        return info

    @classmethod
    def load(cls, path: str) -> pandas.DataFrame:
        """
        加载指定方言读音数据

        Parameters:
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
            comment='#',
            encoding='utf-8'
        )

        data['音標'] = data['音標'].str.translate({
            0x008f: 0x027f, # -> LATIN SMALL LETTER REVERSED R WITH FISHHOOK
            0x0090: 0x0285, # -> LATIN SMALL LETTER SQUAT REVERSED ESH
        })

        return data.replace('', numpy.NAN)

    @functools.lru_cache
    def load_data(self, *ids: tuple[str]) -> pandas.DataFrame:
        logging.info(f'loading data from {self._path} ...')

        paths = self._dialect_info['path'] if len(ids) == 0 \
            else self._dialect_info.loc[pandas.Index(ids), 'path']
        data = []
        for id, p in paths.items():
            logging.info(f'load {p}')

            try:
                d = self.load(p)
            except Exception as e:
                logging.error(f'cannot load file {p}: {e}')
                continue

            d.insert(0, 'did', id)

            # 把原始读音切分成声母、韵母、声调
            seg = d.pop('音標').str.extract(r'([^0-9]*)([0-9][0-9a-z]*)?')
            d[['initial', 'final']] = preprocess.parse(
                preprocess.clean_ipa(seg.iloc[:, 0], force=True)
            ).iloc[:, :2]

            # 汉字音典的原始读音标注的是调号，根据方言详情映射成调值和调类
            tone, cat = self._tone_map[id]
            d['tone'] = seg.iloc[:, 1].map(tone)
            d['tone_category'] = seg.iloc[:, 1].map(cat)

            data.append(d)

        logging.info(f'done, {len(data)} data files loaded.')

        if len(data) == 0:
            return None

        data = pandas.concat(data, axis=0, ignore_index=True)

        if self._normalize:
            # 把读音改写成规范的形式
            data['initial'] = preprocess.normalize_initial(data['initial'])

        # 删除声韵调均为空的记录
        data.dropna(
            how='all',
            subset=['initial', 'final', 'tone'],
            inplace=True
        )

        if self._superscript_tone:
            # 把声调中的普通数字转成上标数字
            data['調值'] = preprocess.tone2super(data['調值'])

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

        if self._uniform_name:
            # 替换列名为统一的名称
            data.rename(columns={'漢字': 'character', '解釋': 'note'}, inplace=True)

        return data

    @property
    def data(self):
        return self.load_data()

    def filter(self, id=None):
        """
        筛选部分数据

        Paramters:
            id (str or tuple): 保留的方言 ID

        Returns:
            data (`sinetym.dataset.Dataset`): 符合条件的数据

        忽略不在数据集中的 ID，如果同一个 ID 在 `id` 中出现了多次，只返回一份数据。
        """

        if id is None:
            id = ()
        else:
            id = sorted(({id} if isinstance(id, str) else set(id)) \
                & set(self._dialect_info.index))
            if len(id) == 0:
                return Dataset(self.name)

        return Dataset(
            self.name,
            data=self.load_data(*id),
            metadata=self.metadata
        )


class XiaoxuetangDataset(Dataset):
    """
    小学堂汉字古今音资料库的现代方言数据集

    支持延迟加载。
    见：https://xiaoxue.iis.sinica.edu.tw/ccr。
    """

    def __init__(
        self,
        path: str,
        normalize: bool = True,
        superscript_tone: bool = False,
        na: str | None = None,
        empty: str | None = None,
        uniform_name: bool = True,
        did_prefix: str | None = 'X',
        cid_prefix: str | None = 'X',
        uniform_info: bool = True
    ):
        """
        Parameters:
            path: 数据集所在的基础路径
            normalize: 为真时，进一步清洗读音数据，改写为规范的形式
            superscript_tone: 为真时，把声调中的普通数字转成上标数字
            na: 代表缺失的字符串，为 None 时保持原状
            empty: 代表零声母/零韵母/零声调的字符串，为 None 时保持原状
            uniform_name: 为真时，把数据列名转为通用名称
            did_prefix: 非空时在方言 ID 添加该前缀
            cid_prefix: 非空时在字 ID 添加该前缀
            uniform_info: 小学堂原始数据为繁体中文，把方言信息转换成简体中文
        """

        super().__init__('xiaoxuetang', metadata={
            'dialect_info': self.load_dialect_info(
                os.path.join(path, 'data', 'csv', 'dialect.csv'),
                did_prefix,
                uniform_name,
                uniform_info
            ),
            'char_info': self.load_char_info(
                os.path.join(path, 'data', 'csv', 'char.csv'),
                cid_prefix,
                uniform_name
            )
        })

        self._path = os.path.join(path, 'data', 'csv', 'dialects')
        self._normalize = normalize
        self._superscript_tone = superscript_tone
        self._na = na
        self._empty = empty
        self._uniform_name = uniform_name
        self._did_prefix = did_prefix
        self._cid_prefix = cid_prefix

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
            subgroup.str.contains('北京|东北|冀鲁|胶辽|中原|兰银|江淮|西南', na=False),
            subgroup.str.replace(
                '.*(北京|东北|冀鲁|胶辽|中原|兰银|江淮|西南).*',
                r'\1官话',
                regex=True
            ),
            numpy.where(
                subgroup.str.contains('闽东|闽南|闽北|闽中|莆仙|邵将|琼文', na=False),
                subgroup.str.replace(
                    '.*(闽东|闽南|闽北|闽中|莆仙|邵将|琼文).*',
                    r'\1区',
                    regex=True
                ),
                numpy.where(
                    subgroup.str.contains('桂南|桂北', na=False),
                    subgroup.str.replace('.*(桂南|桂北).*', r'\1平话', regex=True),
                    numpy.where(
                        subgroup.str.contains('湘南|粤北', na=False),
                        subgroup.str.replace('.*(湘南|粤北).*', r'\1土话', regex=True),
                        ''
                    )
                )
            )
        ), index=subgroup.index).replace('', numpy.NAN)

    @classmethod
    def load_dialect_info(
        cls,
        fname: str,
        did_prefix: str | None = None,
        uniform_name: bool = False,
        uniform_info: bool = False
    ) -> pandas.DataFrame:
        """
        加载方言点信息

        Parameters:
            fname: 方言点信息文件路径
            did_prefix: 非空时在方言 ID 添加该前缀
            uniform_name: 为真时，把数据列名转为通用名称
            uniform_info: 小学堂原始数据为繁体中文，把方言信息转换成简体中文

        Returns:
            info: 方言点信息数据表，只包含文件中编号非空的方言点
        """

        logging.info(f'loading dialect information from {fname}...')
        info = pandas.read_csv(fname, dtype={'編號': str})
        info = info[info['編號'].notna()].set_index('編號')

        info['區'] = cls.clean_subgroup(info['區'])
        # 部分方言点包含来源文献，删除
        info['方言點'] = info['方言點'].str.replace(
            r'\(安徽省志\)|\(珠江三角洲\)|\(客贛方言調查報告\)|\(廣西漢語方言\)'
            r'|\(平話音韻研究\)|\(廣東閩方言\)|\(漢語方音字匯\)|\(當代吳語\)',
            '',
            regex=True
        )

        if did_prefix is not None:
            info.set_index(did_prefix + info.index, inplace=True)

        if uniform_info:
            # 把方言信息转换成简体中文
            info.update(info.select_dtypes(object).map(
                opencc.OpenCC('t2s').convert,
                na_action='ignore'
            ))

            # 少数方言名称转成更通行的名称
            info.replace({'方言': {'客语': '客家话', '其他土话': '土话'}}, inplace=True)

        if uniform_name:
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

    @classmethod
    def load_char_info(
        cls,
        fname: str,
        did_prefix: str | None = None,
        uniform_name: bool = False
    ) -> pandas.DataFrame:
        """
        加载字信息

        Parameters:
            fname: 字信息文件路径
            did_prefix: 非空时在字 ID 添加该前缀
            uniform_name: 为真时，把数据列名转为通用名称

        Returns:
            info: 字信息数据表
        """

        logging.debug(f'load character information from {fname}') 
        info = pandas.read_csv(fname, dtype=str)
        info = info[info['字號'].notna()].set_index('字號')

        if did_prefix is not None:
            info.set_index(did_prefix + info.index, inplace=True)

        if uniform_name:
            info.index.rename('cid', inplace=True)
            info.rename(columns={'字': 'character'}, inplace=True)

        logging.debug(f'{info.shape[0]} dialects loaded.')
        return info

    @classmethod
    def load(cls, path: str) -> pandas.DataFrame:
        """
        加载指定方言读音数据

        Parameters:
            path: 要加载的数据文件路径

        Returns:
            data: 加载的读音数据表
        """

        data = pandas.read_csv(
            path,
            encoding='utf-8',
            dtype=str
        )

        # 清洗数据集特有的错误
        id = os.path.basename(path).partition('.')[0]
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

        return data.replace('', numpy.NAN)

    @functools.lru_cache
    def load_data(self, *ids: tuple[str]) -> pandas.DataFrame:
        """
        加载方言字音数据

        Parameters:
            ids: 要加载的方言 ID 列表，当为空时，加载路径中所有方言数据

        Returns:
            data: 方言字音表
        """

        logging.info(f'loading data from {self._path} ...')

        if len(ids) == 0:
            ids = []
            for e in os.scandir(self._path):
                id, ext = os.path.splitext(e.name)
                if e.is_file() and ext == '.csv':
                    ids.append(id)
            ids = sorted(ids)

            if len(ids) == 0:
                logging.error(f'no data file in {self._path}!')
                return None

        data = []
        for id in ids:
            fname = os.path.join(self._path, id + '.csv')
            logging.info(f'load {fname}')

            try:
                d = self.load(fname)
            except Exception as e:
                logging.error(f'cannot load file {fname}: {e}')
                continue

            d.insert(
                0,
                'did',
                id if self._did_prefix is None else self._did_prefix + id
            )
            data.append(d)

        logging.info(f'done, {len(data)} data files loaded.')

        if len(data) == 0:
            return None

        data = pandas.concat(data, axis=0, ignore_index=True)

        # 清洗读音数据。一个格子可能记录了多个音，用点分隔，只取第一个
        data['聲母'] = preprocess.clean_initial(data['聲母'].str.split('.').str[0])
        data['韻母'] = preprocess.clean_final(data['韻母'].str.split('.').str[0])
        data['調值'] = preprocess.clean_tone(data['調值'].str.split('.').str[0])
        data['調類'] = data['調類'].str.split('.').str[0] \
            .str.replace(r'[^上中下變陰陽平去入輕聲]', '', regex=True)

        if self._normalize:
            # 把读音改写为规范的形式
            data['聲母'] = preprocess.normalize_initial(data['聲母'])

        data.replace(
            {'聲母': '', '韻母': '', '調值': '', '調類': ''},
            numpy.NAN,
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

        if self._uniform_name:
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

            if self._cid_prefix is not None:
                # 字 ID 添加前缀
                data['cid'] = self._cid_prefix + data['cid']

        return data

    @property
    def data(self):
        return self.load_data()

    def filter(self, id: str | list[str] | None = None) -> Dataset:
        """
        筛选部分数据

        Paramters:
            id: 保留的方言 ID

        Returns:
            data: 符合条件的数据
        """

        if id is None:
            id = ()
        elif isinstance(id, str):
            id = (id,)

        return Dataset(
            self.name,
            data=self.load_data(*id),
            metadata=self.metadata
        )

    def clear_cache(self):
        """
        清除已经加载的数据
        """

        self.load.cache_clear()


class ZhongguoyuyanDataset(Dataset):
    """
    中国语言资源保护工程采录展示平台的方言数据集

    支持延迟加载。
    见：https://zhongguoyuyan.cn/。
    """

    def __init__(
        self,
        path: str,
        normalize: bool = True,
        superscript_tone: bool = False,
        na: str | None = None,
        empty: str | None = None,
        uniform_name: bool = True,
        did_prefix: str | None = 'Z',
        cid_prefix: str | None = 'Z'
    ):
        """
        Parameters:
            path: 数据集所在的基础路径
            normalize: 为真时，进一步清洗读音数据，改写为规范的形式
            superscript_tone: 为真时，把声调中的普通数字转成上标数字
            na: 代表缺失数据的字符串，为 None 时保持原状
            empty: 代表零声母/零韵母/零声调的字符串，为 None 时保持原状
            uniform_name: 为真时，把数据列名转为通用名称
            did_prefix: 非空时在方言 ID 添加该前缀
            cid_prefix: 非空时在字 ID 添加该前缀
        """

        super().__init__('zhongguoyuyan', metadata={
            'dialect_info': self.load_dialect_info(
                os.path.join(path, 'csv', 'location.csv'),
                did_prefix,
                uniform_name
            ),
            'char_info': self.load_char_info(
                os.path.join(path, 'csv', 'words.csv'),
                cid_prefix,
                uniform_name
            )
        })

        self._path = os.path.join(path, 'csv', 'dialect')
        self._normalize = normalize
        self._superscript_tone = superscript_tone
        self._na = na
        self._empty = empty
        self._uniform_name = uniform_name
        self._did_prefix = did_prefix
        self._cid_prefix = cid_prefix

    @classmethod
    def clean_location(cls, location: pandas.DataFrame) -> pandas.DataFrame:
        """
        清洗方言点地名格式

        Parameters:
            location: 方言点信息数据表

        Returns:
            clean: 归一化市县名称的方言点数据
        """

        def norm(raw: pandas.Series) -> pandas.Series:
            clean =  raw.str.strip() \
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
                .str.replace('^(.{2,6})[市州盟县区旗]$', r'\1', regex=True) \
                .replace('', numpy.NAN)

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
        clean.loc[mask & clean['county'].isna(), 'county'] = clean['city']
        clean.loc[mask, 'city'] = numpy.NAN

        # 特别行政区市县置空
        clean.loc[
            clean['province'].isin(['香港', '澳门']),
            ['city', 'county']
        ] = numpy.NAN

        return clean

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

        return group.replace('', numpy.NAN)

    @classmethod
    def get_subgroup(cls, location: pandas.DataFrame) -> pandas.Series:
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

        return subgroup.replace('', numpy.NAN)

    @classmethod
    def get_cluster(cls, location: pandas.DataFrame) -> pandas.Series:
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
    def get_subcluster(cls, location: pandas.DataFrame) -> pandas.Series:
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

    @classmethod
    def load_dialect_info(
        cls,
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
        info = cls.clean_location(
            pandas.read_csv(fname, index_col=0)
        )

        # 以地市名加县区名为方言点名称，如地市名和县区名相同，只取其一
        info['spot'] = info['city'].where(
            info['city'] == info['county'],
            info['city'].fillna('') + info['county'].fillna('')
        )
        info['spot'].where(info['spot'] != '', info['province'], inplace=True)

        # 清洗方言区、片、小片名称
        info['group'] = cls.get_group(info)
        info['subgroup'] = cls.get_subgroup(info)
        info['cluster'] = cls.get_cluster(info)
        info['subcluster'] = cls.get_subcluster(info)

        # 个别官话方言点标注的大区和子区不一致，去除
        info.loc[
            (info['group'] == '官话') & ~info['subgroup'].str.endswith('官话', na=False),
            ['group', 'subgroup']
        ] = numpy.NAN

        # 个别方言点的经纬度有误，去除
        info.loc[~info['latitude'].between(0, 55), 'latitude'] = numpy.NAN
        info.loc[~info['longitude'].between(70, 140), 'longitude'] = numpy.NAN

        if did_prefix is not None:
            info.set_index(did_prefix + info.index, inplace=True)

        if uniform_name:
            info.index.rename('did', inplace=True)

        logging.info(f'done, {info.shape[0]} dialects loaded.')
        return info

    @classmethod
    def load_char_info(
        cls,
        fname: str,
        cid_prefix: str | None = None,
        uniform_name: bool = False
    ) -> pandas.DataFrame:
        """
        读取字信息

        Parameters:
            fname: 字信息文件路径
            cid_prefix: 非空时在字 ID 添加该前缀
            uniform_name: 为真时，把数据列名转为通用名称

        Returns:
            info: 字信息数据表
        """

        logging.info(f'loading character information from {fname}...')
        info = pandas.read_csv(fname, dtype=str).set_index('cid')
        
        if cid_prefix is not None:
            info.set_index(cid_prefix + info.index, inplace=True)

        if uniform_name:
            info.rename(columns={'item': 'character'}, inplace=True)

        return info

    @classmethod
    def load(cls, path: str) -> pandas.DataFrame:
        """
        加载指定方言读音数据

        Parameters:
            path: 要加载的数据文件路径

        Returns:
            data: 加载的读音数据表
        """

        data = pandas.read_csv(
            path,
            encoding='utf-8',
            dtype=str
        )

        # 清洗数据集特有的错误
        if os.path.basename(path)[:5] == '02135':
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

        return data.replace('', numpy.NAN)

    @functools.lru_cache
    def load_data(
        self,
        *ids: tuple[str],
        variant: str | None = None
    ) -> pandas.DataFrame:
        """
        加载方言字音数据

        Parameters:
            ids: 要加载的方言列表，当为空时，加载路径中所有方言数据
            variant: 要加载的方言变体，为空时加载所有变体，仅当 ids 为空时生效

        Returns:
            data: 方言字音表

        语保数据文件的编号由3部分组成：<方言 ID><发音人编号><内容编号>，其中：
            - 方言 ID：5个字符
            - 发音人编号：4个字符，代表老年男性、青年男性等
            - 内容编号：2个字符，dz 代表单字音
        """

        logging.info(f'loading data from {self._path} ...')

        if len(ids) == 0:
            ids = sorted([e.name[:9] for e in os.scandir(self._path) \
                if e.is_file() and e.name.endswith('dz.csv') \
                and (variant is None or e.name[5:9] == variant)])

            if len(ids) == 0:
                logging.error(f'no data file in {self._path}!')
                return None

        data = []
        for id in ids:
            fname = os.path.join(self._path, id + 'dz.csv')
            logging.info(f'load {fname}')

            try:
                d = self.load(fname)
            except Exception as e:
                logging.error(f'cannot load file {fname}: {e}')
                continue

            # 添加方言 ID 及变体 ID
            # 语保提供老年男性、青年男性等不同发音人的数据，后几个字符为其编号
            d.insert(
                0,
                'did',
                id[:5] if self._did_prefix is None else self._did_prefix + id[:5]
            )
            d.insert(1, 'variant', id[5:9])
            data.append(d)

        logging.info(f'done, {len(data)} data files loaded.')

        if len(data) == 0:
            return None

        data = pandas.concat(data, axis=0, ignore_index=True)

        # 清洗读音数据
        data['initial'] = preprocess.clean_initial(data['initial'])
        data['finals'] = preprocess.clean_final(data['finals'])
        data['tone'] = preprocess.clean_tone(data['tone'])

        if self._normalize:
            # 把读音改写为规范的形式
            data['initial'] = preprocess.normalize_initial(data['initial'])

        data.replace(
            {'initial': '', 'finals': '', 'tone': ''},
            numpy.NAN,
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

        if self._uniform_name:
            # 替换列名为统一的名称
            data.rename(columns={
                'iid': 'cid',
                'name': 'character',
                'finals': 'final',
                'memo': 'note'
            }, inplace=True)

            if self._cid_prefix is not None:
                # 字 ID 添加前缀
                data['cid'] = self._cid_prefix + data['cid']

        return data

    @property
    def data(self):
        return self.load_data()

    def filter(
        self,
        id: str | list[str] | None = None,
        variant: str | None = None
    ) -> Dataset:
        """
        筛选部分数据

        Paramters:
            id: 保留的方言记录 ID
            variant: 保留的变体代号，仅当未指定 id 时生效

        Returns:
            data: 符合条件的数据
        """

        if id is None:
            id = ()
        elif isinstance(id, str):
            id = (id,)

        return Dataset(
            self.name,
            data=self.load_data(*id, variant=variant),
            metadata=self.metadata
        )

    def clear_cache(self):
        """
        清除已经加载的数据
        """

        self.load.cache_clear()


try:
    path = os.environ['MCPDICT_HOME']
except KeyError:
    logging.error(
        'Environment variable MCPDICT_HOME not set! Set this variable to '
        'MCPDict\'s home directory to make use of the dataset.'
    )
else:
    mcpdict = MCPDictDataset(path)

path = os.environ.get('XIAOXUETANG_HOME')
if path is None:
    logging.error('Set variable environment XIAOXUETANG_HOME then reload this module.')
else:
    xiaoxuetang = XiaoxuetangDataset(path)

path = os.environ.get('ZHONGGUOYUYAN_HOME')
if path is None:
    logging.error('Set variable environment ZHONGGUOYUYAN_HOME then reload this module.')
else:
    zhongguoyuyan = ZhongguoyuyanDataset(path)
