from time import time
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
# 下面这一行用于初始化cuda环境，不可删除
import pycuda.autoinit

# from .utils import func_time

G_LOGGER = trt.Logger(trt.Logger.INFO)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class LayoutInfer(object):
    def __init__(self, filepath, allocate_shapes) -> None:
        self.allocate_shapes = allocate_shapes
        self.engine = self.loadEngine2TensorRT(filepath)
        self.context = self.engine.create_execution_context()
        self.context.active_optimization_profile = 0  # 这句话必写
        self.inputs, self.outputs, self.bindings = self.allocate_buffers(
            self.engine, True, allocate_shapes)
        self.stream = cuda.Stream()
        self.input_num, self.output_num = len(self.inputs), len(self.outputs)

    def loadEngine2TensorRT(self, filepath):
        '''
        通过加载计划文件，构建TensorRT运行引擎
        '''
        G_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(G_LOGGER)
        # 反序列化引擎
        with open(filepath, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def allocate_buffers(self, engine, is_explicit_batch=False, shapes=None):
        '''
        显存申请
        shapes: 所有输入输出所对应的shape，如：[(1,3,1024,1024), (1,1,1024,1024)]
        '''
        inputs = []
        outputs = []
        bindings = []

        index = 0
        for binding in engine:
            dims = engine.get_binding_shape(binding)

            assert(shapes is not None)
            for i in range(len(shapes[index])):
                dims[i] = shapes[index][i]
            index += 1

            # The maximum batch size which can be used for inference.
            size = trt.volume(dims) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            # host_mem = cuda.pagelocked_empty(size, np.float32)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Determine whether a binding is an input binding.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        # print("t1\n", inputs)
        # print("t2\n", outputs)
        return inputs, outputs, bindings

    def pre_process(self, inputs):
        raise NotImplementedError

    def post_process(self, outputs):
        raise NotImplementedError

    def set_inputshape(self, inputs):
        '''
        设置动态输入的shape
        '''
        for i in range(self.input_num):
            self.context.set_binding_shape(i, inputs[i].shape)

    def set_inputhost(self, inputs):
        for i in range(self.input_num):
            self.inputs[i].host = inputs[i]

    def get_outputhost(self):
        outputs = []
        for i in range(self.output_num):
            outputs.append(self.outputs[i].host[0:np.array(self.context.get_binding_shape(
                self.input_num + i)).prod()].reshape(self.context.get_binding_shape(self.input_num + i)))
        return outputs

    def layout_infer(self, hidden_states, attention_mask, rel_pos):
        assert (self.engine is not None)
        # self.context.active_optimization_profile = 0  # 这句话必写

        self.set_inputshape([hidden_states, attention_mask, rel_pos])

        self.set_inputhost([hidden_states, attention_mask, rel_pos])
        # self.set_outputshape(self.allocate_shapes)
        # Copy from the Python buffer src to the device pointer dest (an int or a DeviceAllocation) asynchronously,
        [cuda.memcpy_htod_async(inp.device, inp.host.ravel(), self.stream)
         for inp in self.inputs]
        # Wait for all activity on this stream to cease, then return.
        # Asynchronously execute inference on a batch.
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle)
        # Copy from the device pointer src (an int or a DeviceAllocation) to the Python buffer dest asynchronously
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
         for out in self.outputs]
        self.stream.synchronize()
        outputs = self.get_outputhost()
        return outputs

