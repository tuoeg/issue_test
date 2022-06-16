"""
============================
# -*- coding: utf-8 -*-
# @Time    : 2022/6/16 16:29
# @Author  : Yingjie Bai
# @FileName: TRT_issue.py
===========================
"""
import os
import tensorrt as trt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np

from Layout.trt_infer import LayoutInfer

# Dataloader
bbox = np.load('./new_data/bbox.npy')[20:26].astype(np.int32)
images = np.load('./new_data/images.npy')[20:26].astype(np.float32)
input_ids = np.load('./new_data/input_ids.npy')[20:26].astype(np.int32)


def infer():

    Layout_dyn_trt = LayoutInfer("./dyn_test.plan",
                        [[6, 512], [6, 512, 4], [6, 3, 224, 224], [6, 709, 768], [6, 709, 768]])
    Layout_sta_trt = LayoutInfer("./static_test.plan",
                        [[6, 512], [6, 512, 4], [6, 3, 224, 224], [6, 709, 768], [6, 709, 768]])

    out_dyn = Layout_dyn_trt.layout_infer(input_ids, bbox, images)
    out_sta = Layout_sta_trt.layout_infer(input_ids, bbox, images)

    # Loading the results of Torch
    o1 = np.load('./out1.npy')
    o2 = np.load('./out2.npy')

    # Compare
    import traceback

    try:
        np.testing.assert_almost_equal(o2, out_dyn[0], 5)
    except:
        print(traceback.print_exc())

    try:
        np.testing.assert_almost_equal(o1, out_dyn[1], 5)
    except:
        print(traceback.print_exc())

    try:
        np.testing.assert_almost_equal(o2, out_sta[0], 5)
    except:
        print(traceback.print_exc())

    try:
        np.testing.assert_almost_equal(o1, out_sta[1], 5)
    except:
        print(traceback.print_exc())

if __name__ == '__main__':
    infer()