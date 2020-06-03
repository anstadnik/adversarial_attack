import unittest
from text_item import TextItem
from img import Img
from img_ocr import ImgOCR
import numpy as np

class TestTextItem(unittest.TestCase):

    """This class tests TextItem"""

    def test_all(self):
        img = np.random.uniform(size=(10, 10))
        for left in [0, 10, 100]:
            for top in [0, 10, 100]:
                for width in [0, 10, 100]:
                    for height in [0, 10, 100]:
                        for conf in [0., 10., 100.]:
                            for text in ['lorem', 'ipsum', 'dolor']:
                                with self.subTest():
                                    ti = TextItem(left, top, width, height, conf, text)
                                    self.assertIsNotNone(ti)
                                    self.assertIsNone(ti.noise)
                                    with self.assertRaises(RuntimeError):
                                        ti.img
                                    ti.img = img
                                    ti.noise = img
                                    self.assertIsNotNone(ti.noise)
                                    self.assertIsNotNone(ti.img)

class TestImg(unittest.TestCase):

    """This class tests Img"""

    def test_img(self):
        img = np.random.uniform(size=(10, 10))
        img = Img(img=img)
        self.assertIsNotNone(img)

    def test_load(self):
        img = Img(path='1590850515.png')
        self.assertIsNotNone(img)

class TestImgOcr(unittest.TestCase):

    """This class tests Img"""

    def setUp(self):
        self.img = ImgOCR(path='1590850515.png')

    def test_compute_data(self):
        self.assertIsNotNone(self.img.compute_text_data())

    def test_get_str(self):
        self.assertIsNotNone(self.img.get_str())

class TestImgOcrTextItem(unittest.TestCase):
    def setUp(self):
        self.img = ImgOCR(path='1590850515.png')
        self.data = self.img.compute_text_data()

    def test_get_img(self):
        for d in self.data:
            self.assertIsNotNone(d.img)
        
    def test_get_conf(self):
        for d in self.data:
            self.assertIsNotNone(d.conf)

    def test_get_text(self):
        for d in self.data:
            self.assertIsNotNone(d.text)

class TestHiding(unittest.TestCase):
    def setUp(self):
        self.img = ImgOCR(path='1590850515.png')
        self.data = self.img.compute_text_data()

    def test_hiding0(self):
        d = self.data[0]
        self.img.hide_data(d)
        data = self.img.compute_text_data()
        for d_ in data:
            self.assertNotEqual(d, d_)

    def test_hiding1(self):
        d = self.data[1]
        self.img.hide_data(d)
        data = self.img.compute_text_data()
        for d_ in data:
            self.assertNotEqual(d, d_)

    def test_hiding2(self):
        d = self.data[2]
        self.img.hide_data(d)
        data = self.img.compute_text_data()
        for d_ in data:
            self.assertNotEqual(d, d_)

    def test_hiding3(self):
        d = self.data[3]
        self.img.hide_data(d)
        data = self.img.compute_text_data()
        for d_ in data:
            self.assertNotEqual(d, d_)

    def test_hiding4(self):
        d = self.data[4]
        self.img.hide_data(d)
        data = self.img.compute_text_data()
        for d_ in data:
            self.assertNotEqual(d, d_)

    def test_hiding5(self):
        d = self.data[5]
        self.img.hide_data(d)
        data = self.img.compute_text_data()
        for d_ in data:
            self.assertNotEqual(d, d_)
