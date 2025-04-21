# -*- coding: utf-8 -*-

"""
测试 SinComp 数据集相关功能
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import json
import os
import pandas
import selenium.webdriver
import tempfile
import unittest
import unittest.mock
import urllib

import sincomp.datasets


data_dir = os.path.join(os.path.dirname(__file__), 'data')


def mock_urlopen(url, *args, **kwargs):
    """
    模拟网络请求，返回测试用数据
    """

    if isinstance(url, urllib.request.Request):
        url = url.full_url

    return open(
        os.path.join(
            data_dir,
            urllib.parse.urlparse(url).path.split('/')[-1]
        ),
        'rb'
    )

class MockChrome(unittest.mock.MagicMock):
    """
    模拟 selenium Chrome 浏览器驱动，代替真正的浏览器和网络请求，返回测试用数据
    """

    def get(self, url):
        self.components = urllib.parse.urlparse(url)

    def get_log(self, log_type):
        if self.components.path == '/index':
            url = self.components \
                ._replace(path='/api/mongo/query/latestSurveyMongo').geturl()
        elif self.components.path.startswith('/point/'):
            url = self.components \
                ._replace(path='/api/mongo/resource/normal').geturl()
        else:
            url = self.components.geturl()

        message = {
            'method': 'Network.responseReceived',
            'params': {
                'requestId': '1.1',
                'response': {
                    'url': url
                }
            }
        }
        return [{'message': json.dumps({'message': message})}]

    def execute_cdp_cmd(self, cmd, cmd_args):
        fname = 'survey.json' if self.components.path == '/index' \
            else 'standard.json' if self.components.path == '/api/api/media/standard' \
            else self.components.path.split('/')[-1] + '.json'

        with open(
            os.path.join(data_dir, 'zhongguoyuyan', fname),
            encoding='utf-8'
        ) as f:
            return {'body': f.read()}


def setUpModule():
    """
    使用模拟函数取代真正的网络请求
    """

    global urlopen_patcher
    global chrome_patcher

    urlopen_patcher = unittest.mock.patch.object(
        urllib.request,
        'urlopen',
        mock_urlopen
    )
    chrome_patcher = unittest.mock.patch.object(
        selenium.webdriver,
        'Chrome',
        MockChrome
    )

    urlopen_patcher.start()
    chrome_patcher.start()

    import importlib
    import sincomp.datasets
    importlib.reload(sincomp.datasets)

def tearDownModule():
    chrome_patcher.stop()
    urlopen_patcher.stop()

    import importlib
    import sincomp.datasets
    importlib.reload(sincomp.datasets)


class TestDataset(unittest.TestCase):
    def setUp(self):
        super().setUp()

        data = []
        prefix = os.path.join(data_dir, 'custom_dataset1')
        for fname in '08533.csv', '23C57.csv':
            data.append(pandas.read_csv(
                os.path.join(prefix, fname),
                encoding='utf-8',
                dtype=str
            ))

        self.dataset = sincomp.datasets.Dataset(
            pandas.concat(data, axis=0, ignore_index=True)
        )

    def test_get_data(self):
        data = self.dataset.get_data('08533')
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 20)
        self.assertListEqual(
            data.columns.tolist(),
            ['did', 'character', 'cid', 'initial', 'final', 'tone']
        )

    def test_dialect_ids(self):
        self.assertListEqual(self.dataset.dialect_ids, ['08533', '23C57'])

    def test_dialects(self):
        dialects = self.dataset.dialects
        self.assertIsInstance(dialects, pandas.DataFrame)
        self.assertListEqual(dialects.index.tolist(), ['08533', '23C57'])

    def test_characters(self):
        chars = self.dataset.characters
        self.assertIsInstance(chars, pandas.DataFrame)
        self.assertGreater(chars.shape[0], 0)
        self.assertIn('character', chars.columns)

    def test_data(self):
        data = self.dataset.data
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 47)
        self.assertListEqual(
            data.columns.tolist(),
            ['did', 'character', 'cid', 'initial', 'final', 'tone']
        )

    def test_items(self):
        count = 0
        for did, data in self.dataset.items():
            self.assertIsInstance(data, pandas.DataFrame)
            self.assertListEqual(
                data.columns.tolist(),
                ['did', 'character', 'cid', 'initial', 'final', 'tone']
            )
            self.assertTrue((data['did'] == did).all())

            count += 1

        self.assertEqual(count, 2)

    def test_iterrows(self):
        count = 0
        for i, r in self.dataset.iterrows():
            self.assertIsInstance(r, pandas.Series)
            self.assertListEqual(
                r.index.tolist(),
                ['did', 'character', 'cid', 'initial', 'final', 'tone']
            )

            count += 1

        self.assertEqual(count, self.dataset.data.shape[0])

    def test_select(self):
        other = self.dataset.select(['08533'])
        self.assertIsInstance(other, sincomp.datasets.LinkDataset)
        self.assertListEqual(other.dialect_ids, ['08533'])

        data = other.data
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 20)
        self.assertListEqual(
            data.columns.tolist(),
            ['did', 'character', 'cid', 'initial', 'final', 'tone']
        )

    def test_sample(self):
        other = self.dataset.sample(n=1)
        self.assertIsInstance(other, sincomp.datasets.LinkDataset)
        self.assertEqual(len(other), 1)
        self.assertIn(other.dialect_ids[0], self.dataset.dialect_ids)

        data = other.data
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertListEqual(
            data.columns.tolist(),
            ['did', 'character', 'cid', 'initial', 'final', 'tone']
        )

    def test_shuffle(self):
        other = self.dataset.shuffle()
        self.assertIsInstance(other, sincomp.datasets.LinkDataset)
        self.assertSetEqual(set(self.dataset.dialect_ids), set(other.dialect_ids))

        data1 = self.dataset.data
        data2 = other.select().data
        self.assertIsInstance(data2, pandas.DataFrame)
        self.assertEqual(data1.shape[0], data2.shape[0])
        self.assertListEqual(data1.columns.tolist(), data2.columns.tolist())

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_iter(self):
        count = 0
        for data in iter(self.dataset):
            self.assertIsInstance(data, pandas.DataFrame)
            self.assertListEqual(
                data.columns.tolist(),
                ['did', 'character', 'cid', 'initial', 'final', 'tone']
            )

            count += 1

        self.assertEqual(count, 2)

    def test_getitem(self):
        data = self.dataset['08533']
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 20)
        self.assertListEqual(
            data.columns.tolist(),
            ['did', 'character', 'cid', 'initial', 'final', 'tone']
        )

    def test_getitem_list(self):
        data = self.dataset[['08533']].data
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 20)
        self.assertListEqual(
            data.columns.tolist(),
            ['did', 'character', 'cid', 'initial', 'final', 'tone']
        )

    def test_add(self):
        other = sincomp.datasets.FileDataset(
            data_dir=os.path.join(data_dir, 'custom_dataset2')
        )
        output = self.dataset + other

        self.assertIsInstance(output, sincomp.datasets.LinkDataset)
        self.assertEqual(len(output), len(self.dataset) + len(other))

        data = output.data
        self.assertEqual(
            data.shape[0],
            self.dataset.data.shape[0] + other.data.shape[0]
        )
        self.assertListEqual(
            data.columns.tolist(),
            ['did', 'character', 'cid', 'initial', 'final', 'tone']
        )


class TestFileDataset(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.dataset = sincomp.datasets.FileDataset(
            data_dir=os.path.join(data_dir, 'custom_dataset1')
        )

    def test_get_data(self):
        data = self.dataset.get_data('08533')
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 20)
        self.assertListEqual(
            data.columns.tolist(),
            ['did', 'character', 'cid', 'initial', 'final', 'tone']
        )

    def test_dialect_ids(self):
        self.assertListEqual(self.dataset.dialect_ids, ['08533', '23C57'])

    def test_dialects(self):
        dialects = self.dataset.dialects
        self.assertIsInstance(dialects, pandas.DataFrame)
        self.assertListEqual(dialects.index.tolist(), ['08533', '23C57'])

    def test_characters(self):
        chars = self.dataset.characters
        self.assertIsInstance(chars, pandas.DataFrame)
        self.assertGreater(chars.shape[0], 0)
        self.assertIn('character', chars.columns)

    def test_data(self):
        data = self.dataset.data
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 47)
        self.assertListEqual(
            data.columns.tolist(),
            ['did', 'character', 'cid', 'initial', 'final', 'tone']
        )


class TestCCRDataset(unittest.TestCase):
    def setUp(self):
        """
        使用测试用方言信息文件代替正式的文件
        """

        super().setUp()

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.dataset = sincomp.datasets.CCRDataset(
            self.tmp_dir.name,
            dialect_file=os.path.join(data_dir, 'ccr_dialects.csv')
        )

    def tearDown(self):
        super().tearDown()
        self.tmp_dir.cleanup()

    def test_get_data(self):
        data = self.dataset.get_data('027')
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 20)
        self.assertListEqual(
            data.columns.tolist(),
            [
                'did',
                'cid',
                'character',
                'initial',
                'final',
                'tone',
                'tone_category',
                'note'
            ]
        )

    def test_dialect_ids(self):
        self.assertListEqual(self.dataset.dialect_ids, ['027', '072'])

    def test_dialects(self):
        dialects = self.dataset.dialects
        self.assertIsInstance(dialects, pandas.DataFrame)
        self.assertListEqual(dialects.index.tolist(), ['027', '072'])
        self.assertListEqual(
            dialects.columns.tolist(),
            [
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
            ]
        )

    def test_characters(self):
        chars = self.dataset.characters
        self.assertIsInstance(chars, pandas.DataFrame)
        self.assertGreater(chars.shape[0], 0)
        self.assertIn('character', chars.columns)

    def test_data(self):
        data = self.dataset.data
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 40)
        self.assertListEqual(
            data.columns.tolist(),
            [
                'did',
                'cid',
                'character',
                'initial',
                'final',
                'tone',
                'tone_category',
                'note'
            ]
        )


class TestMCPDictDataset(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.dataset = sincomp.datasets.MCPDictDataset(self.tmp_dir.name)

    def tearDown(self):
        super().tearDown()
        self.tmp_dir.cleanup()

    def test_tone_map(self):
        tone_map = self.dataset.tone_map
        self.assertIsInstance(tone_map, dict)
        self.assertSetEqual(set(tone_map.keys()), {'無極', '趙縣'})
        self.assertEqual(len(tone_map['無極']), 2)
        self.assertIsInstance(tone_map['無極'][0], dict)

    def test_get_data(self):
        data = self.dataset.get_data('無極')
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 20)
        self.assertListEqual(
            data.columns.tolist(),
            [
                'did',
                'character',
                'initial',
                'final',
                'tone',
                'tone_category',
                'note'
            ]
        )

    def test_dialect_ids(self):
        self.assertListEqual(self.dataset.dialect_ids, ['無極', '趙縣'])

    def test_dialects(self):
        dialects = self.dataset.dialects
        self.assertIsInstance(dialects, pandas.DataFrame)
        self.assertListEqual(dialects.index.tolist(), ['無極', '趙縣'])
        self.assertListEqual(
            dialects.columns.tolist(),
            [
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
            ]
        )

    def test_characters(self):
        chars = self.dataset.characters
        self.assertIsInstance(chars, pandas.DataFrame)
        self.assertGreater(chars.shape[0], 0)
        self.assertIn('character', chars.columns)

    def test_data(self):
        data = self.dataset.data
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 40)
        self.assertListEqual(
            data.columns.tolist(),
            [
                'did',
                'character',
                'initial',
                'final',
                'tone',
                'tone_category',
                'note'
            ]
        )


class TestZhongguoyuyanDataset(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.dataset = sincomp.datasets.ZhongguoyuyanDataset(
            self.tmp_dir.name,
            downloader_kwargs={'delay': None}
        )

    def tearDown(self):
        super().tearDown()
        self.tmp_dir.cleanup()

    def test_get_dialects(self):
        dialects = self.dataset.get_dialects()
        self.assertIsInstance(dialects, pandas.DataFrame)
        self.assertEqual(dialects.shape[0], 2)
        self.assertListEqual(
            dialects.columns.tolist(),
            [
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
            ]
        )

    def test_get_data(self):
        data = self.dataset.get_data('06K06')
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 20)
        self.assertListEqual(
            data.columns.tolist(),
            ['did', 'cid', 'character', 'initial', 'final', 'tone', 'note']
        )

    def test_dialect_ids(self):
        self.assertLessEqual(self.dataset.dialect_ids, ['06K06', '06K10'])

    def test_dialects(self):
        dialects = self.dataset.dialects
        self.assertIsInstance(dialects, pandas.DataFrame)
        self.assertEqual(dialects.shape[0], 2)
        self.assertListEqual(
            dialects.columns.tolist(),
            [
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
            ]
        )

    def test_characters(self):
        chars = self.dataset.characters
        self.assertIsInstance(chars, pandas.DataFrame)
        self.assertGreater(chars.shape[0], 0)
        self.assertIn('character', chars.columns)

    def test_data(self):
        data = self.dataset.data
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 40)
        self.assertListEqual(
            data.columns.tolist(),
            ['did', 'cid', 'character', 'initial', 'final', 'tone', 'note']
        )


class TestDatasets(unittest.TestCase):
    def test_list_datasets(self):
        self.assertListEqual(sincomp.datasets.list_datasets(), ['CCR', 'MCPDict', 'zhongguoyuyan'])

    def test_get(self):
        self.assertIsInstance(
            sincomp.datasets.get(
                os.path.join(data_dir, 'custom_dataset1', '23C57.csv')
            ),
            sincomp.datasets.Dataset
        )
        self.assertIsInstance(
            sincomp.datasets.get(os.path.join(data_dir, 'custom_dataset1')),
            sincomp.datasets.FileDataset
        )
        self.assertIsInstance(
            sincomp.datasets.get('CCR'),
            sincomp.datasets.CCRDataset
        )
        self.assertIsInstance(
            sincomp.datasets.get('MCPDict'),
            sincomp.datasets.MCPDictDataset
        )
        self.assertIsInstance(
            sincomp.datasets.get('zhongguoyuyan'),
            sincomp.datasets.ZhongguoyuyanDataset
        )
        self.assertIs(sincomp.datasets.get('foo'), None)