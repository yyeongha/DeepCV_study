from PIL import ImageFilter
import random


class TwoCropsTransform:
    """하나의 이미지에서 두 개의 랜덤 크롭을 가져와 쿼리와 키로 사용"""
    # -> 이는 논문에서 언급한 두개의 증강된 이미지 뷰를 만들어내는 과정

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x): # 입력이미지 x를 받아서 두개의 증강된 이미지 뷰를 만들어내는 함수
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """SimCLR에서 사용된 가우시안 블러 증강 https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
