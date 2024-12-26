# -*- coding: utf-8 -*-

"""
测试 SinComp 数据集相关功能
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import os
import pandas
import unittest
import unittest.mock

from common import data_dir, setUpModule, tearDownModule


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        import sincomp.datasets

        data = []
        prefix = os.path.join(data_dir, 'custom_dataset1')
        for fname in os.listdir(prefix):
            data.append(pandas.read_csv(
                os.path.join(prefix, fname),
                encoding='utf-8',
                dtype=str
            ))

        cls.dataset = sincomp.datasets.Dataset(
            pandas.concat(data, axis=0, ignore_index=True)
        )

    def test_data(self):
        data = self.dataset.data
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertGreater(data.shape[0], 0)
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            self.assertIn(col, data.columns)

    def test_dialects(self):
        dialects = self.dataset.dialects
        self.assertIsInstance(dialects, pandas.DataFrame)
        self.assertSetEqual(set(dialects.index), {'08533', '23C57'})

    def test_characters(self):
        chars = self.dataset.characters
        self.assertIsInstance(chars, pandas.DataFrame)
        self.assertGreater(chars.shape[0], 0)
        self.assertIn('character', chars.columns)

    def test_dialect_ids(self):
        self.assertSetEqual(set(self.dataset.dialect_ids), {'08533', '23C57'})

    def test_items(self):
        count = 0
        for did, data in self.dataset.items():
            self.assertIsInstance(data, pandas.DataFrame)
            for col in 'did', 'cid', 'initial', 'final', 'tone':
                self.assertIn(col, data.columns)

            self.assertTrue((data['did'] == did).all())

            count += 1

        self.assertEqual(count, 2)

    def test_filter(self):
        data = self.dataset.filter(['08533']).data
        self.assertIsInstance(data, pandas.DataFrame)
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            self.assertIn(col, data.columns)

    def test_iter(self):
        count = 0
        for data in iter(self.dataset):
            self.assertIsInstance(data, pandas.DataFrame)
            for col in 'did', 'cid', 'initial', 'final', 'tone':
                self.assertIn(col, data.columns)

            count += 1

        self.assertEqual(count, 2)


class TestFileDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        import sincomp.datasets
        cls.dataset = sincomp.datasets.FileDataset(
            path=os.path.join(data_dir, 'custom_dataset1')
        )

    def test_file_dataset(self):
        self.assertEqual(len(self.dataset), 2)

    def test_dialect_ids(self):
        self.assertSetEqual(set(self.dataset.dialect_ids), {'08533', '23C57'})

    def test_items(self):
        count = 0
        for did, data in self.dataset.items():
            self.assertIsInstance(data, pandas.DataFrame)
            for col in 'did', 'cid', 'initial', 'final', 'tone':
                self.assertIn(col, data.columns)

            self.assertTrue((data['did'] == did).all())

            count += 1

        self.assertEqual(count, 2)

    def test_data(self):
        data = self.dataset.data
        self.assertIsInstance(data, pandas.DataFrame)
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            self.assertIn(col, data.columns)

    def test_data_empty(self):
        import sincomp.datasets

        dataset = sincomp.datasets.FileDataset(
            path=os.path.join(data_dir, 'not_exist')
        )
        data = dataset.data

        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 0)
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            self.assertIn(col, data.columns)

    def test_iterrows(self):
        count = 0
        for i, r in self.dataset.iterrows():
            self.assertIsInstance(r, pandas.Series)
            for col in 'did', 'cid', 'initial', 'final', 'tone':
                self.assertIn(col, r)

            count += 1

        self.assertEqual(count, self.dataset.data.shape[0])

    def test_filter(self):
        data = self.dataset.filter(['08533']).data
        self.assertIsInstance(data, pandas.DataFrame)
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            self.assertIn(col, data.columns)

    def test_sample(self):
        data = self.dataset.sample(n=1).data
        self.assertIsInstance(data, pandas.DataFrame)
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            self.assertIn(col, data.columns)

    def test_shuffle(self):
        data = self.dataset.shuffle().data
        self.assertIsInstance(data, pandas.DataFrame)
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            self.assertIn(col, data.columns)

    def test_append(self):
        import sincomp.datasets

        other = sincomp.datasets.FileDataset(
            path=os.path.join(data_dir, 'custom_dataset2')
        )
        output = self.dataset.append(other)

        self.assertIsInstance(output, sincomp.datasets.FileDataset)
        self.assertEqual(len(output), len(self.dataset) + len(other))
        self.assertEqual(
            output.data.shape[0],
            self.dataset.data.shape[0] + other.shape[0]
        )


class TestCCRDataset(unittest.TestCase):
    def test_dialects(self):
        import sincomp.datasets

        dialects = sincomp.datasets.get('CCR').dialects
        self.assertIsInstance(dialects, pandas.DataFrame)
        self.assertGreater(dialects.shape[0], 0)
        self.assertSetEqual(
            set(dialects.columns),
            {
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
            }
        )

    def test_characters(self):
        import sincomp.datasets

        chars = sincomp.datasets.get('CCR').characters
        self.assertIsInstance(chars, pandas.DataFrame)
        self.assertGreater(chars.shape[0], 0)
        self.assertIn('character', chars.columns)

    def test_load_data(self):
        import sincomp.datasets

        _, data = sincomp.datasets.get('CCR').load_data('027')[0]
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 20)
        for col in 'did', 'cid', 'character', 'initial', 'final', 'tone':
            self.assertIn(col, data.columns)

class TestMCPDictDataset(unittest.TestCase):
    def test_dialects(self):
        import sincomp.datasets

        dialects = sincomp.datasets.get('MCPDict').dialects
        self.assertIsInstance(dialects, pandas.DataFrame)
        self.assertEqual(dialects.shape[0], 2)
        self.assertSetEqual(
            set(dialects.columns),
            {
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
            }
        )

    def test_characters(self):
        import sincomp.datasets

        chars = sincomp.datasets.get('MCPDict').characters
        self.assertIsInstance(chars, pandas.DataFrame)
        self.assertEqual(chars.shape[0], 0)
        self.assertIn('character', chars.columns)

    def test_data(self):
        import sincomp.datasets

        data = sincomp.datasets.get('MCPDict').data
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 40)
        for col in 'did', 'character', 'initial', 'final', 'tone':
            self.assertIn(col, data.columns)

    def test_refresh(self):
        import sincomp.datasets
        sincomp.datasets.get('MCPDict').refresh()


class TestZhongguoyuyanDataset(unittest.TestCase):
    def test_dialects(self):
        import sincomp.datasets

        dialects = sincomp.datasets.get('zhongguoyuyan').dialects
        self.assertIsInstance(dialects, pandas.DataFrame)
        self.assertEqual(dialects.shape[0], 2)
        self.assertSetEqual(
            set(dialects.columns),
            {
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
            }
        )

    def test_characters(self):
        import sincomp.datasets

        chars = sincomp.datasets.get('zhongguoyuyan').characters
        self.assertIsInstance(chars, pandas.DataFrame)
        self.assertGreater(chars.shape[0], 0)
        self.assertIn('character', chars.columns)

    def test_data(self):
        import sincomp.datasets

        data = sincomp.datasets.get('zhongguoyuyan').data
        self.assertIsInstance(data, pandas.DataFrame)
        self.assertEqual(data.shape[0], 40)
        for col in 'did', 'character', 'initial', 'final', 'tone':
            self.assertIn(col, data.columns)

    def test_refresh(self):
        import sincomp.datasets
        sincomp.datasets.get('zhongguoyuyan').refresh()


class TestDatasets(unittest.TestCase):
    def test_get(self):
        import sincomp.datasets

        for name in (
            'CCR',
            'ccr',
            'xiaoxue',
            'MCPDict',
            'mcpdict',
            'zhongguoyuyan',
            'yubao',
            os.path.join(data_dir, 'custom_dataset1'),
            os.path.join(data_dir, 'custom_dataset1', '23C57.csv')
        ):
            self.assertIsInstance(
                sincomp.datasets.get(name),
                sincomp.datasets.Dataset
            )

        self.assertIs(sincomp.datasets.get('foo'), None)
