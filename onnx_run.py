import onnxruntime
import numpy as np

# 动态 onnx
sess_dyn = onnxruntime.InferenceSession('./dyn_test.onnx')
# 静态 onnx
sess_sta = onnxruntime.InferenceSession('./static_test.onnx')

# Dataloader
bbox = np.load('./new_data/bbox.npy')[20:26].astype(np.int32)
images = np.load('./new_data/images.npy')[20:26].astype(np.float32)
input_ids = np.load('./new_data/input_ids.npy')[20:26].astype(np.int32)

# Start Inference
res_dyn = sess_dyn.run([], {"input_ids": input_ids, "bbox": bbox, "images": images})

res_sta = sess_sta.run([], {"input_ids": input_ids, "bbox": bbox, "images": images})

# Loading the results of Torch
o1 = np.load('./out1.npy')
o2 = np.load('./out2.npy')


# Compare
np.testing.assert_almost_equal(res_dyn[0], o1, 5)
np.testing.assert_almost_equal(res_dyn[1], o2, 5)

np.testing.assert_almost_equal(res_sta[0], o1, 5)
np.testing.assert_almost_equal(res_sta[1], o2, 5)