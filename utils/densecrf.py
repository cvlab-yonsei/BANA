import numpy as np
import pydensecrf.densecrf as DCRF
from pydensecrf.utils import unary_from_softmax

class DENSE_CRF(object):
    def __init__(self, bi_w, bi_xy_std, bi_rgb_std, pos_w=3, pos_xy_std=3, max_iter=10):
        self.bi_w, self.bi_xy_std, self.bi_rgb_std = bi_w, bi_xy_std, bi_rgb_std
        self.pos_w, self.pos_xy_std = pos_w, pos_xy_std
        self.max_iter = max_iter

    def inference(self, image, prob):
        '''
        image : array (HxWx3)
        unary : array (CxHxW)
        '''
        C, H, W = prob.shape
        img = np.ascontiguousarray(image)
        unary = unary_from_softmax(prob)
        unary = np.ascontiguousarray(unary)
        
        d = DCRF.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=img, compat=self.bi_w)
        
        Q = d.inference(self.max_iter)
        out = np.array(Q).reshape((C, H, W))
        return out