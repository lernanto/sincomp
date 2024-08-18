# -*- coding: utf-8 -*-

"""
汉语方言读音数据集

当前支持读取：
    - 汉字音典的现代方言数据，见：https://mcpdict.sourceforge.io/
    - 小学堂汉字古今音资料库的现代方言数据，见：https://xiaoxue.iis.sinica.edu.tw/ccrdata/
    - 中国语言资源保护工程采录展示平台的方言数据，见：https://zhongguoyuyan.cn/
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import sys
import os
import logging
import pandas
import numpy
import retry
import io
import urllib.request
import urllib.error
import zipfile
import json
import re
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
        data: pandas.DataFrame | None = None,
        name: str | None = None
    ):
        """
        Parameters:
            data: 方言字音数据表
            name: 数据集名字
        """

        self._data = data
        self.name = name

    @property
    def data(self) -> pandas.DataFrame | None:
        return self._data

    def __iter__(self):
        data = self.data
        return iter(()) if data is None else iter(data)

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
        name: str | None = None
    ):
        """
        Parameters:
            file_map: 方言 ID 到数据文件路径的映射表
            path: 数据集所在的目录路径
        """

        if file_map is None and path is not None:
            # 未指定数据文件映射表但指定了数据目录，把目录下每个文件看成一个方言点，主文件名为方言 ID
            file_map = pandas.Series(*zip(
                *[(os.path.join(c, f), os.path.splitext(f)[0]) \
                    for c, _, fs in os.walk(path) for f in fs]
            ))

        if name is None and path is not None:
            name = os.path.basename(os.path.abspath(path))

        super().__init__(name=name)
        self._file_map = file_map

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

        return self.load_file(self._file_map[did])

    def items(self):
        """
        依次访问每个方言点的数据
        """

        for did in self._file_map.index:
            yield did, self.load(did)

    def __iter__(self):
        for _, data in self.items():
            yield data

    @property
    def data(self) -> pandas.DataFrame:
        """
        返回所有方言读音数据

        Returns:
            output: 合并所有文件数据的长表
        """

        output = pandas.concat(self, axis=0, ignore_index=True)

        logging.debug(
            f'{self._file_map.shape[0]} dialects '
            f'{output.shape[0]} records loaded.'
        )
        return output

    def iterrows(self):
        """
        依次访问每各方言点的每条记录
        """

        for data in self:
            for r in data.iterrows():
                yield r

    def filter(self, idx) -> Dataset:
        """
        从数据集中筛选满足条件的方言

        只支持针对方言筛选。

        Parameters:
            idx: 任何 pandas.Series 接受的索引或筛选条件

        Returns:
            output: 筛选后的数据集，只包含满足条件的方言
        """

        return FileDataset(file_map=self._file_map.loc[idx])

    def sample(self, *args, **kwargs) -> Dataset:
        """
        从数据集随机抽样部分方言

        Parameters:
            args, kwargs: 透传给 pandas.Serires.sample 用于抽样方言

        Returns:
            output: 包含抽样方言的数据集
        """

        return FileDataset(file_map=self._file_map.sample(*args, **kwargs))

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

        return FileDataset(file_map=self._file_map.sample(
            frac=1.0,
            random_state=random_state
        ))

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

        return self._file_map.shape[0]

    def __add__(self, other: Dataset) -> Dataset:
        return self.append(other)


class FileCacheDataset(FileDataset):
    """
    使用本地文件作为缓存的数据集

    首次加载一个方言数据时，会加载原始数据并处理成 FileDataset 能处理的格式，保存在本地缓存文件。
    以后加载会加载缓存文件，以加快加载速度。

    TODO: 对 FileCacheDataset 执行数据集操作如 sample、append 的结果会退化成 FileDataset，
    因此访问结果数据时会直接读取缓存文件，而不管该文件是否存在。为避免这种情况需要增加复杂的处理逻辑，
    为保持代码简洁，不特殊处理，而是由使用者保证在执行上述操作前生成所有缓存文件。
    """

    def __init__(self, cache_dir: str, dids: list[str], name: str | None = None):
        """
        Parameters:
            cache_dir: 缓存文件所在目录路径
            dids: 数据集包含的所有方言 ID 列表
        """

        super().__init__(
            cache_dir + os.sep + pandas.Series(dids, index=dids),
            name=name
        )
        self._cache_dir = cache_dir

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

        # 如果已存在缓存文件，直接读取
        if os.path.isfile(self._file_map[did]):
            logging.info(f'using cache file {self._file_map[did]}.')
            return super().load_file(self._file_map[did])

        # 不存在缓存文件，从原始数据读取
        # 因为原始数据可能一次不止加载一个方言，因此返回的是方言 ID 和数据表的列表
        data_list = self.load_data(did)
        # 写入文件缓存
        for i, d in data_list:
            if not os.path.isfile(self._file_map[i]):
                logging.info(f'create cache file {self._file_map[i]}.')
                os.makedirs(self._cache_dir, exist_ok=True)
                d.to_csv(
                    self._file_map[i],
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

        for path in self._file_map:
            try:
                logging.info(f'remove cache file {path}.')
                os.remove(path)
            except FileNotFoundError:
                ...
            except OSError as e:
                logging.warning(e)

        # 删除缓存目录
        try:
            logging.info(f'remove cache directory {self._cache_dir}.')
            os.rmdir(self._cache_dir)
        except OSError as e:
            logging.warning(e)

    def refresh(self):
        """
        强制更新全部缓存文件
        """

        self.clear_cache()
        for did in self._file_map.index:
            self.load(did)


class MCPDictDataset(FileCacheDataset):
    """
    汉字音典方言数据集

    见：https://mcpdict.sourceforge.io/。
    """

    def __init__(
        self,
        cache_dir: str,
        uniform_name: bool = True,
        uniform_info: bool = True,
        did_prefix: str | None = None,
        cid_prefix: str | None = None,
        superscript_tone: bool = False,
        na: str | None = None,
        empty: str | None = '∅',
        name: str = 'MCPDict'
    ):
        """
        Parameters:
            cache_dir: 缓存文件所在目录路径
            uniform_name: 为真时，把数据列名转为通用名称
            uniform_info: 小学堂原始数据为繁体中文，把方言信息转换成简体中文
            did_prefix: 非空时在方言 ID 添加该前缀
            cid_prefix: 非空时在字 ID 添加该前缀
            superscript_tone: 为真时，把声调中的普通数字转成上标数字
            na: 代表缺失的字符串，为 None 时保持原状
            empty: 代表零声母/零韵母/零声调的字符串，为 None 时保持原状
            name: 数据集名称
        """

        self._path = os.path.join(cache_dir, 'tools', 'tables', 'output')
        self._uniform_name = uniform_name
        self._uniform_info = uniform_info
        self._did_prefix = did_prefix
        self._cid_prefix = cid_prefix
        self._superscript_tone = superscript_tone
        self._na = na
        self._empty = empty

        if not os.path.isdir(self._path):
            # 数据文件不存在，先从汉字音典项目页面下载数据
            logging.info(
                'run for the first time, download data from Web, '
                'this may take a while.'
            )
            self.download(cache_dir)

        info = self.load_dialect_info()
        super().__init__(cache_dir, info.index, name=name)

        # 从方言详情提取声调调值和调类的映射表
        self._tone_map = {}
        for i, m in info.pop('聲調').map(json.loads).items():
            tone = {}
            cat = {}
            for k, v in m.items():
                # 少数连读声调有特殊符号，暂时去除
                tone[k] = re.sub(f'[^{"".join(preprocess._TONES)}]', '', v[0])
                cat[k] = v[3]
            self._tone_map[i] = tone, cat

        self.dialect_info = info
        self.metadata = {'dialect_info': info}

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

        logging.info(f'downloading {url}...')

        with urllib.request.urlopen(url) as res:
            with zipfile.ZipFile(io.BytesIO(res.read())) as zf:
                os.makedirs(os.path.join(output, 'tools', 'tables', 'output'))
                logging.info(f'extracting files to {output}...')

                for info in zf.infolist():
                    # 路径第一段是带版本号的项目名，需去除
                    path = info.filename.partition('/')[2]
                    # 把字音数据目录的所有文件解压到目标路径
                    if not info.is_dir() and path.startswith('tools/tables/output/'):
                        logging.info(f'extracting {info.filename}...')
                        path = os.path.join(*[output] + path.split('/'))
                        with open(path, 'wb') as of:
                            of.write(zf.read(info))

        logging.info('done.')

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
            logging.warning(f'no valid data in {self._path}.')
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
    def load_raw(cls, id: str, path: str) -> pandas.DataFrame:
        """
        从原始文件加载指定方言读音数据

        Parameters:
            id: 要加载的方言的原始 ID（未加前缀）
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

        return data.replace('', numpy.nan)

    def load_data(self, did: str) -> list[tuple[str, pandas.DataFrame]]:
        """
        加载方言读音数据并清理

        Parameters:
            did: 要加载的方言 ID

        Returns:
            did: 同参数 `did`
            data: 方言读音数据表
        """

        logging.info(f'load data from {self.dialect_info.at[did, "path"]}.')
        data = self.load_raw(
            did if self._did_prefix is None else did[len(self._did_prefix):],
            self.dialect_info.at[did, 'path']
        )
        data['did'] = did

        # 把原始读音切分成声母、韵母、声调
        seg = data.pop('音標').str.extract(r'([^0-9]*)([0-9][0-9a-z]*)?')
        data[['initial', 'final']] = preprocess.parse(
            preprocess.clean_ipa(seg.iloc[:, 0], force=True)
        ).iloc[:, :2]

        # 汉字音典的原始读音标注的是调号，根据方言详情映射成调值和调类
        tone, cat = self._tone_map[did]
        data['tone'] = seg.iloc[:, 1].map(tone)
        data['tone_category'] = seg.iloc[:, 1].map(cat)

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

        return ((did, data),)


class CCRDataset(FileCacheDataset):
    """
    小学堂汉字古今音资料库的现代方言数据集

    见：https://xiaoxue.iis.sinica.edu.tw/ccrdata/。
    """

    def __init__(
        self,
        cache_dir: str,
        uniform_name: bool = True,
        uniform_info: bool = True,
        did_prefix: str | None = None,
        cid_prefix: str | None = None,
        superscript_tone: bool = False,
        na: str | None = None,
        empty: str | None = None,
        name: str = 'CCR'
    ):
        """
        Parameters:
            cache_dir: 缓存文件所在目录路径
            uniform_name: 为真时，把数据列名转为通用名称
            uniform_info: 小学堂原始数据为繁体中文，把方言信息转换成简体中文
            did_prefix: 非空时在方言 ID 添加该前缀
            cid_prefix: 非空时在字 ID 添加该前缀
            superscript_tone: 为真时，把声调中的普通数字转成上标数字
            na: 代表缺失的字符串，为 None 时保持原状
            empty: 代表零声母/零韵母/零声调的字符串，为 None 时保持原状
            name: 数据集名称
        """

        self._path = cache_dir
        self._uniform_name = uniform_name
        self._uniform_info = uniform_info
        self._did_prefix = did_prefix
        self._cid_prefix = cid_prefix
        self._superscript_tone = superscript_tone
        self._na = na
        self._empty = empty

        info = self.load_dialect_info()
        super().__init__(cache_dir, info.index, name=name)
        self.dialect_info = info
        self.metadata = {
            'dialect_info': info,
            'char_info': self.load_char_info()
        }

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

    def load_dialect_info(self) -> pandas.DataFrame:
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

        info = pandas.read_csv(
            os.path.join(os.path.dirname(__file__), 'ccr_dialect_info.csv'),
            dtype={'編號': str}
        ).dropna(subset=['編號'])

        # 各方言数据下载地址
        info['url'] = 'https://xiaoxue.iis.sinica.edu.tw/ccrdata/file/' \
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
        })
        # 下载后解压的文件路径
        info['path'] = self._path + os.sep + info['編號'] + ' ' \
            + info['方言'].where(info['方言'] != '官話', info['區']) + \
            '_' + info['方言點'] + '.xlsx'
        info.set_index('編號', inplace=True)

        info['區'] = self.clean_subgroup(info['區'])
        # 部分方言点包含来源文献，删除
        info['方言點'] = info['方言點'].str.replace(
            r'\(安徽省志\)|\(珠江三角洲\)|\(客贛方言調查報告\)|\(廣西漢語方言\)'
            r'|\(平話音韻研究\)|\(廣東閩方言\)|\(漢語方音字匯\)|\(當代吳語\)',
            '',
            regex=True
        )

        if self._did_prefix is not None:
            info.set_index(self._did_prefix + info.index, inplace=True)

        if self._uniform_info:
            # 把方言信息转换成简体中文
            info.update(info[['方言', '區', '片／小區', '小片', '方言點']].map(
                opencc.OpenCC('t2s').convert,
                na_action='ignore'
            ))

            # 少数方言名称转成更通行的名称
            info.replace({'方言': {'客语': '客家话', '其他土话': '土话'}}, inplace=True)

        if self._uniform_name:
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

        return info

    def load_char_info(self) -> pandas.DataFrame:
        """
        加载字信息

        Returns:
            info: 字信息数据表
        """

        info = pandas.read_csv(
            os.path.join(os.path.dirname(__file__), 'ccr_char_info.csv'),
            dtype=str
        )
        info = info[info['字號'].notna()].set_index('字號')

        if self._did_prefix is not None:
            info.set_index(self._did_prefix + info.index, inplace=True)

        if self._uniform_name:
            info.index.rename('cid', inplace=True)
            info.rename(columns={'字': 'character'}, inplace=True)

        return info

    @staticmethod
    @retry.retry(exceptions=urllib.error.URLError, tries=3, delay=1)
    def download(url: str, output: str) -> None:
        """
        从小学堂网站下载方言读音数据

        Parameters:
            url: 下载地址
            output: 保存下载解压文件的本地目录
        """

        logging.info(f'downloading {url}...')

        # 设置 User-Agent，否则请求会被拒绝
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as res:
            with zipfile.ZipFile(io.BytesIO(res.read())) as zf:
                logging.info(f'extracting files to {output}...')
                os.makedirs(output, exist_ok=True)

                for info in zf.infolist():
                    # 压缩包路径编码为 Big5，但 zipfile 默认用 CP437 解码，需重新用 Big5 解码
                    try:
                        fname = info.filename.encode('cp437').decode('big5')
                    except UnicodeError:
                        fname = info.filename
                    # 改正文件名中的别字
                    fname = fname.replace('閔', '閩')

                    logging.info(f'extracting {fname}...')
                    with open(os.path.join(output, fname), 'wb') as of:
                        of.write(zf.read(info))

        logging.info('done.')

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

        return data.replace('', numpy.nan)

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

        path = self.dialect_info.at[did, 'path']
        if not os.path.isfile(path):
            # 方言数据文件不存在，从网站下载
            self.download(self.dialect_info.at[did, 'url'], self._cache_dir)

        logging.info(f'loading data from {path}...')
        data = self.load_raw(
            did if self._did_prefix is None else did[len(self._did_prefix):],
            path
        )
        data['did'] = did

        # 清洗读音数据。一个格子可能记录了多个音，用点分隔，只取第一个
        data['聲母'] = preprocess.clean_initial(data['聲母'].str.split('.').str[0])
        data['韻母'] = preprocess.clean_final(data['韻母'].str.split('.').str[0])
        data['調值'] = preprocess.clean_tone(data['調值'].str.split('.').str[0])
        data['調類'] = data['調類'].str.split('.').str[0] \
            .str.replace(r'[^上中下變陰陽平去入輕聲]', '', regex=True)

        data.replace(
            {'聲母': '', '韻母': '', '調值': '', '調類': ''},
            numpy.nan,
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

        return ((did, data),)


class ZhongguoyuyanDataset(FileCacheDataset):
    """
    中国语言资源保护工程采录展示平台的方言数据集

    见：https://zhongguoyuyan.cn/。
    """

    def __init__(
        self,
        cache_dir: str,
        path: str,
        uniform_name: bool = True,
        did_prefix: str | None = None,
        cid_prefix: str | None = None,
        superscript_tone: bool = False,
        na: str | None = None,
        empty: str | None = None,
        name: str = 'zhongguoyuyan'
    ):
        """
        Parameters:
            cache_dir: 缓存文件所在目录路径
            path: 数据集所在的基础路径
            uniform_name: 为真时，把数据列名转为通用名称
            did_prefix: 非空时在方言 ID 添加该前缀
            cid_prefix: 非空时在字 ID 添加该前缀
            superscript_tone: 为真时，把声调中的普通数字转成上标数字
            na: 代表缺失数据的字符串，为 None 时保持原状
            empty: 代表零声母/零韵母/零声调的字符串，为 None 时保持原状
            name: 数据集名称
        """

        self._path = os.path.join(path, 'csv')
        self._uniform_name = uniform_name
        self._did_prefix = did_prefix
        self._cid_prefix = cid_prefix
        self._superscript_tone = superscript_tone
        self._na = na
        self._empty = empty

        info = self.load_dialect_info()
        super().__init__(cache_dir, info.index, name=name)
        self.dialect_info = info
        self.metadata = {
            'dialect_info': info,
            'char_info': self.load_char_info()
        }

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
                .replace('', numpy.nan)

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
        clean.loc[mask, 'city'] = numpy.nan

        # 特别行政区市县置空
        clean.loc[
            clean['province'].isin(['香港', '澳门']),
            ['city', 'county']
        ] = numpy.nan

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

        return group.replace('', numpy.nan)

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

        return subgroup.replace('', numpy.nan)

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

        使用规则归一化原始数据中的市县名称，以及方言大区、子区名称等信息。

        Returns:
            info: 方言点信息数据表
        """

        info = pandas.read_csv(
            os.path.join(self._path, 'location.csv'),
            index_col=0
        )
        info['path'] = os.path.join(self._path, 'dialect') + os.sep + \
            info.index + 'mb01dz.csv'

        info = self.clean_location(info)

        # 以地市名加县区名为方言点名称，如地市名和县区名相同，只取其一
        info['spot'] = info['city'].where(
            info['city'] == info['county'],
            info['city'].fillna('') + info['county'].fillna('')
        )
        info['spot'] = info['spot'].where(info['spot'] != '', info['province'])

        # 清洗方言区、片、小片名称
        info['group'] = self.get_group(info)
        info['subgroup'] = self.get_subgroup(info)
        info['cluster'] = self.get_cluster(info)
        info['subcluster'] = self.get_subcluster(info)

        # 个别官话方言点标注的大区和子区不一致，去除
        info.loc[
            (info['group'] == '官话') & ~info['subgroup'].str.endswith('官话', na=False),
            ['group', 'subgroup']
        ] = numpy.nan

        # 个别方言点的经纬度有误，去除
        info.loc[~info['latitude'].between(0, 55), 'latitude'] = numpy.nan
        info.loc[~info['longitude'].between(70, 140), 'longitude'] = numpy.nan

        if self._did_prefix is not None:
            info.set_index(self._did_prefix + info.index, inplace=True)

        if self._uniform_name:
            info.index.rename('did', inplace=True)

        return info

    def load_char_info(self) -> pandas.DataFrame:
        """
        读取字信息

        Returns:
            info: 字信息数据表
        """

        path = os.path.join(self._path, 'words.csv')
        info = pandas.read_csv(path, dtype=str).set_index('cid')
        
        if self._cid_prefix is not None:
            info.set_index(self._cid_prefix + info.index, inplace=True)

        if self._uniform_name:
            info.rename(columns={'item': 'character'}, inplace=True)

        return info

    @classmethod
    def load_raw(cls, id: str, path: str) -> pandas.DataFrame:
        """
        加载指定方言读音数据

        Parameters:
            id: 要加载的方言的原始 ID（未加前缀）
            path: 要加载的数据文件路径

        Returns:
            data: 加载的读音数据表

        语保数据文件的编号由3部分组成：<方言 ID><发音人编号><内容编号>，其中：
            - 方言 ID：5个字符
            - 发音人编号：4个字符，代表老年男性、青年男性等
            - 内容编号：2个字符，dz 代表单字音
        """

        data = pandas.read_csv(
            path,
            encoding='utf-8',
            dtype=str
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

        return data.replace('', numpy.nan)

    def load_data(self, did: str) -> list[tuple[str, pandas.DataFrame]]:
        """
        加载方言字音数据

        Parameters:
            ids: 要加载的方言列表，当为空时，加载路径中所有方言数据

        Returns:
            data: 方言字音表
        """

        data = self.load_raw(
            did if self._did_prefix is None else did[len(self._did_prefix):],
            self.dialect_info.loc[did, 'path']
        )
        data['did'] = did

        # 清洗读音数据
        data['initial'] = preprocess.clean_initial(data['initial'])
        data['finals'] = preprocess.clean_final(data['finals'])
        data['tone'] = preprocess.clean_tone(data['tone'])

        data.replace(
            {'initial': '', 'finals': '', 'tone': ''},
            numpy.nan,
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

        return ((did, data),)


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

try:
    mcpdict = MCPDictDataset(os.path.join(cache_dir, 'mcpdict'))
except Exception as e:
    logging.error(e)

try:
    path = os.environ['ZHONGGUOYUYAN_HOME']
except KeyError:
    logging.warning(
        'Set environment variable ZHONGGUOYUYAN_HOME to zhongguoyuyan\'s home '
        'dirctory then reload this module to make use of the dataset.'
    )
else:
    zhongguoyuyan = ZhongguoyuyanDataset(
        os.path.join(cache_dir, 'zhongguoyuyan'),
        path
    )

_datasets = {
    'ccr': ccr,
    'CCR': ccr,
    'mcpdict': mcpdict,
    'MCPDict': mcpdict,
    'zhongguoyuyan': zhongguoyuyan
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
        logging.info(f'{name} is not a predefined dataset, try loading data from files.')

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
            logging.warning(f'{name} not found!')


if __name__ == '__main__':
    # 刷新所有数据集的缓存文件
    print('refresh cache files for all datasets. this may take a while.')
    for name in 'mcpdict', 'ccr', 'zhongguoyuyan':
        dataset = globals().get(name)
        if dataset is not None:
            print(f'refreshing {name}...')
            dataset.refresh()
            print(f'done writing {dataset.dialect_info.shape[0]} dialects.')
