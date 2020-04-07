"""
This module contains ImgOCR class
"""
from dataclasses import astuple

import cv2
import pixcat
import pytesseract
from PIL import Image
from tqdm import tqdm

from text_item import TextItem
from img import Img

class ImgOCR(Img):
    """This class is capable of interacting with pytesseract"""

    def __init__(self, *, img=None, path=None):
        """Initialize the class

        Args:
            path (str, optional): path to the image
        """
        super().__init__(img=img, path=path)
        self.data = None
        self.string = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.data is None:
                return
            for v in self.data:
                l, t, w, h, _, text, _ = astuple(v)
                if l <= x <= l + w and t <= y <= t + h:
                    print(text)

    def compute_text_data(self, filter_data=True, add_img=False):
        """Get list of TextItem from pytesseract for the image

        This functions provides data about the text on the image (it's bounding boxes,
        text, confidences etc)

        Args:
            img (): input image
        """
        img = Image.fromarray(self.img)
        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT)
        # 'level',# Page, block, paragraph, line, word
        # 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
        # 'left', 'top', 'width', 'height', 'conf', 'text',

        # Convert dict of lists to list of dicts
        data = [dict(zip(data, t)) for t in zip(*data.values())]

        # Build TextItem objects
        keys = ['left', 'top', 'width', 'height', 'conf', 'text']
        self.data = [TextItem(**{k:d[k] for k in keys}) for d in data]

        if filter_data:
            self.data = list(
                filter(lambda i: i.conf != -1 and i.text != '', self.data))
        if add_img:
            for v in tqdm(self.data, leave=False):
                l, t, w, h = v.left, v.top, v.width, v.height
                v.img = self.img[t:t+h, l:l+w]
        return self.data

    def get_str(self):
        """Generate string from image

        This function extracts text from the image
        """
        img = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        self.string = pytesseract.image_to_string(img)
        return self.string

    def show(self, *, annotate=False, method=None):
        """Display the image

        This function displays images in a various ways

        Args:
            annotate (bool, optional): Set to true to display image with additional data
            method (string, optional): kitty, jupyter or None
        """
        img = self.__annotate(show_text=True) if annotate else self.img
        super().show(img=img, method=method)


    def __annotate(self, *, show_text=False, show_confidence=False):
        """This function annotates the image with bounding boxes and textes or confidences

        Args:
            show_text (bool, optional): set to True to show text
            show_confidence (bool, optional): set to True to show confidence

        """
        img_ = self.img.copy()
        for v in self.data:
            l, t, w, h, conf, text, _ = astuple(v)
            cv2.rectangle(img_, (l, t), (l+w, t+h), (0, 0, 0), -2)
            text_to_show = ""
            if show_text:
                text_to_show += text
            if show_confidence:
                text_to_show += conf
            cv2.putText(img_, str(text_to_show), (l+4, t+7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1)
        return img_
