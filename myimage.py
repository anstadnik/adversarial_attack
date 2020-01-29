"""This module handles the image processing"""
import cv2
# import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
import pixcat


class MyImage():
    """This class is capable of storing the image and interacting with pytesseract"""

    def __init__(self, path=None, resize=None, *, kitty=None, jupyter=None):
        """Initialize the class

        If resize argument is float, it specifies the scaling percent. If it's a tuple,
        it specifies the dimensions of the resulting image. Additionally, if the 2-nd parameter
        equals -1, it is calculated so the proportions of the image are preserved.

        Args:
            path (str, optional): path to the image
            resize (float or tuple, optional): look at the description
            kitty (bool): set to True to display images in kitty terminal
        """
        self.img = None
        self.kitty = kitty
        self.jupyter = jupyter
        self.data = None
        self.string = None

        if path:
            self.get_img(path)
            if resize is not None:
                self.__resize(resize)

        if kitty is not None:
            self.kitty = kitty


    def get_img(self, path: str):
        """Read image from the disk"""
        self.img = cv2.imread(path)


    def show(self, *, annotate=False, method=None):
        """Display the image

        This function displays images in a various ways

        Args:
            annotate (bool, optional): Set to true to display image with additional data
            method (string, optional): kitty, jupyter or None
        """
        if annotate:
            img = self.__annotate(show_text=True)
        else:
            img = self.img
        if method == "kitty" or self.kitty:
            pixcat.Image(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).\
                thumbnail(512).show(align="left")
        elif method == "jupyter" or self.jupyter:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            cv2.imshow('image', img)
            while cv2.waitKey(0) != 27:
                continue
            cv2.destroyAllWindows()


    def compute_data(self, filter_data=False):
        """Get data from pytesseract for the image

        This functions provides data about the text on the image (it's bounding boxes,
        text, confidences etc)

        Args:
            img (): input image
        """
        #  TODO: Check for rotation later <20-01-20, astadnik> #
        img = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        self.data = pd.DataFrame(data)
        if filter_data:
            self.__filter_data()
        return self.data

    def get_str(self):
        """Generate string from image

        This function extracts text from the image

        Args:
            image (): input image
        """
        #  TODO: Check for rotation later <20-01-20, astadnik> #
        img = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        self.string = pytesseract.image_to_string(img)
        return self.string

    def __annotate(self, *, show_text=False, show_confidence=False):
        """This function annotates the image with bounding boxes and textes or confidences

        Args:
            show_text (bool, optional): set to True to show text
            show_confidence (bool, optional): set to True to show confidence

        """
        img_ = self.img.copy()
        for _, row in self.data.iterrows():
            l, t, w, h, conf, text = row['left':'text']
            cv2.rectangle(img_, (l, t), (l+w, t+h), (0, 0, 0), -2)
            text_to_show = ""
            if show_text:
                text_to_show += text
            if show_confidence:
                text_to_show += conf
            cv2.putText(img_, str(text_to_show), (l+4, t+7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255),  1)
        return img_

    def __filter_data(self):
        """Remove empty records from the dataframe

        This function removes the empty records (ones with only whitespace characters)

        """
        self.data = self.data.drop(self.data[self.data['conf'] == '-1'].index)
        self.data = self.data.drop(self.data[self.data['text'].str.strip() == ''].index)
        self.data.reset_index(inplace=True)
        return self.data

    def __resize(self, resize):
        """Handy function to resize image
        """
        if isinstance(resize, float):
            assert 0 < resize <= 1
            width = int(self.img.shape[1] * resize)
            height = int(self.img.shape[0] * resize)
            dim = (width, height)
        elif isinstance(resize, (tuple, list)):
            if resize[1] == -1:
                dim = (resize[0], self.img.shape[0] * resize[0] // self.img.shape[1])
            else:
                dim = resize
        else:
            raise RuntimeError("Wrong resize argument: " + str(resize))
        self.img = cv2.resize(self.img, dim, interpolation=cv2.INTER_AREA)
