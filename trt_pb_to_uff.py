import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
#import tensorflow as tf

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common


# /usr/src/tensorrt/samples/python/introductory_parser_samples


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_save_engine(model_file):

    # For more information on TRT basics, refer to the introductory samples.
    engine_file_path='Body.engine'

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:

        builder.max_workspace_size  = 1 << 30
        
        # default FP32
        # FP16
        #builder.fp16_mode = True

        # Parse the Uff Network
        parser.register_input("input_1", [3, 224, 224], order=trt.UffInputOrder.NCHW)
        parser.register_output("fc3/Softmax")
        parser.parse(model_file, network)

        print('parser done')

        # Build and return an engine.
        with builder.build_cuda_engine(network) as engine:
       
            print('engine done')

            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())


if __name__=='__main__':


    build_save_engine(model_file='./model.uff')


