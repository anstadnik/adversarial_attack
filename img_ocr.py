"""
This module contains ImgOCR class
"""
from dataclasses import astuple
from typing import List

import cv2
import pixcat
import pytesseract
from PIL import Image
from tqdm import tqdm

from text_item import TextItem
from img import Img
import numpy as np
from attack import gen_noise
from multiprocessing import Pool
from imageio import imwrite

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
        self.data_to_process = None
        self.tqdm = None

    def hide_data(self, d):
        l, t, w, h, _, text, _, _, _ = astuple(d)
        img_with_noise = gen_noise((d.img.astype(np.float) / 255) - 0.5)
        if img_with_noise:
            img_with_noise = (img_with_noise[0] + 0.5) * 255
            self.img[t:t+h, l:l+w] = img_with_noise
        

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.data is None:
                return
            for d in self.data:
                l, t, w, h, _, text, _, _, _ = astuple(d)
                if l <= x <= l + w and t <= y <= t + h:
                    self.hide_data(d)
                    self.compute_text_data()
                    img = self.__annotate(show_text=True)
                    self.update(img=img)

    def compute_text_data(self, filter_data=True, add_img=True) -> List[TextItem]:
        """Get list of TextItem from pytesseract for the image

        This functions provides data about the text on the image (it's bounding boxes,
        text, confidences etc)

        Args:
            img (): input image
        """
        # print('Computing data...')
        img = Image.fromarray(self.img)
        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT, config=f'--psm 6 --oem 0')
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
        # print('Data is computed')
        return self.data

    def get_str(self):
        """Generate string from image

        This function extracts text from the image
        """
        img = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        self.string = pytesseract.image_to_string(img, config=f'--psm 6 --oem 0')
        return self.string

    def show(self, *, annotate=False, method=None):
        """Display the image

        This function displays images in a various ways

        Args:
            annotate (bool, optional): Set to true to display image with additional data
            method (string, optional): kitty, jupyter or None
        """
        img = self.__annotate(show_text=True) if annotate else self.img

        if img is None:
            return None
        if method == "kitty":
            pixcat.Image(Image.fromarray(img)).thumbnail(
                512).show(align="left")
        elif method == "jupyter":
            return Image.fromarray(img)
        else:
            cv2.namedWindow('image')
            cv2.setMouseCallback('image', self.mouse_callback)
            cv2.imshow('image', img)
            # wait = p.apply_async(cv2.waitKey, args=(0, ))
            # key = None
            while True:
                # if wait.ready():
                #     key = wait.get()
                    # wait = p.apply_async(cv2.waitKey, args=(0, ))
                key = cv2.waitKey(1)

                if key == 27:
                    break
                if key == 97:
                    annotate = not annotate
                    img = self.__annotate(show_text=True) if annotate else self.img
                    cv2.imshow('image', img)
                elif key == 114:
                    self.compute_text_data()
                    img = self.__annotate(show_text=True) if annotate else self.img
                    self.update(img=img)
                elif key == 115:
                    imwrite('changed_img.png', self.img)
                elif key == 112 and not self.data_to_process:
                    self.data_to_process = self.compute_text_data()
                    self.tqdm = tqdm(total=len(self.data_to_process))

                if self.data_to_process:
                    d = self.data_to_process.pop()
                    self.tqdm.update()
                    if not self.data_to_process:
                        self.data_to_process = None
                        self.tqdm = None
                    if d.conf < 10:
                        continue

                    l, t, w, h, _, text, _, _, _ = astuple(d)
                    img_with_noise = gen_noise((d.img.astype(np.float) / 255) - 0.5)
                    if img_with_noise:
                        img_with_noise = (img_with_noise[0] + 0.5) * 255
                        self.img[t:t+h, l:l+w] = img_with_noise
                    else:
                        print('SHIIIIT')

                    self.compute_text_data()
                    img = self.__annotate(show_text=True)
                    cv2.imshow('image', img)


            cv2.destroyAllWindows()
        return None

    def process(self, pop_size_=None, mutation_rate_=None):
        for d in tqdm(self.compute_text_data()):
            if d.conf < 10:
                continue
            l, t, w, h, _, text, _, _, _ = astuple(d)
            img_with_noise = gen_noise((d.img.astype(np.float) / 255) - 0.5,
                    pop_size_, mutation_rate_)
            if img_with_noise:
                img_with_noise = (img_with_noise[0] + 0.5) * 255
                self.img[t:t+h, l:l+w] = img_with_noise
            else:
                print('SHIIIIT')
        
    def __annotate(self, *, show_text=False, show_confidence=False):
        """This function annotates the image with bounding boxes and textes or confidences

        Args:
            show_text (bool, optional): set to True to show text
            show_confidence (bool, optional): set to True to show confidence

        """
        img_ = self.img.copy()
        for v in self.data:
            l, t, w, h, conf, text, _, _, _ = astuple(v)
            cv2.rectangle(img_, (l, t), (l+w, t+h), (0, 0, 0), -2)
            text_to_show = ""
            if show_text:
                text_to_show += text
            if show_confidence:
                text_to_show += conf
            font_scale = (w * h) / (img_.shape[0] * img_.shape[1])
            cv2.putText(img_, str(text_to_show), (l+2, t+h-4), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 1)
        return img_
