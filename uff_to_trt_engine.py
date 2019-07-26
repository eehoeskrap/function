
'''
need to python path !!!!!!!!!!
1. /host_temp/
2. /usr/src/tensorrt/samples/python/introductory_parser_samples

'''
import random
from PIL import Image
import numpy as np

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt
#import tensorflow as tf

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class ModelData(object):

	MODEL_PATH = "/host_temp/pose-estimation-trt/data/CPN_TENSORRT_FP16.uff"
	INPUT_NAME = "tower_0/Placeholder"
	INPUT_SHAPE = (32, 256, 192, 3)
	OUTPUT_NAME = "tower_0/refine_out/BatchNorm/FusedBatchNorm"
	# We can convert TensorRT data types to numpy types with trt.nptype()
	DTYPE = trt.float32

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

'''
# Allocate host and device buffers, and create a stream.
def allocate_buffers(engine):
	# Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
	h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
	h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
	# Allocate device memory for inputs and outputs.
	d_input = cuda.mem_alloc(h_input.nbytes)
	d_output = cuda.mem_alloc(h_output.nbytes)
	# Create a stream in which to copy inputs/outputs and run inference.
	stream = cuda.Stream()
	return h_input, d_input, h_output, d_output, stream

def do_inference(context, h_input, d_input, h_output, d_output, stream):
	# Transfer input data to the GPU.
	cuda.memcpy_htod_async(d_input, h_input, stream)
	# Run inference.
	context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
	# Transfer predictions back from the GPU.
	cuda.memcpy_dtoh_async(h_output, d_output, stream)
	# Synchronize the stream
	stream.synchronize()

# The UFF path is used for TensorFlow models. You can convert a frozen TensorFlow graph to UFF using the included convert-to-uff utility.
def build_engine_uff(model_file):
	# You can set the logger severity higher to suppress messages (or lower to display more messages).
	with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
		# Workspace size is the maximum amount of memory available to the builder while building an engine.
		# It should generally be set as high as possible.
		builder.max_workspace_size = 1 << 20 
		# We need to manually register the input and output nodes for UFF.
		parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
		parser.register_output(ModelData.OUTPUT_NAME)
		# Load the UFF model and parse it in order to populate the TensorRT network.
		parser.parse(model_file, network)
		# Build and return an engine.
		return builder.build_cuda_engine(network)

def load_normalized_test_case(test_image, pagelocked_buffer):
	# Converts the input image to a CHW Numpy array
	def normalize_image(image):
		# Resize, antialias and transpose the image to CHW.
		c, h, w = ModelData.INPUT_SHAPE
		return np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()

	# Normalize the image and copy to pagelocked memory.
	np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
	return test_image
'''
def main():
	# Set the data path to the directory that contains the trained models and test images for inference.
	#data_path, data_files = common.find_sample_data(description="Runs a ResNet50 network with a TensorRT inference engine.", subfolder="resnet50", find_files=["binoculars.jpeg", "reflex_camera.jpeg", "tabby_tiger_cat.jpg", ModelData.MODEL_PATH, "class_labels.txt"])
	# Get test images, models and labels.
	#test_images = data_files[0:3]

	#uff_model_file, labels_file = data_files[3:]
	#labels = open(labels_file, 'r').read().split('\n')

	#uff_model_file = "/host_temp/pose-estimation-trt/data/CPN_TENSORRT_FP16.uff"



	
	#model_file = "/host_temp/pose-estimation-trt/data/CPN_TENSORRT_FP16.uff"
	#model_file = "/host_temp/pose-estimation-trt/data/CPN_TENSORFLOW_FROZEN.uff"
	
	model_file = "/usr/src/tensorrt/samples/python/introductory_parser_samples/CPN_TENSORFLOW_FROZEN.uff"
	with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
		parser.register_input("tower_0/Placeholder", (32, 256, 192, 3))

		parser.register_output("tower_0/refine_out/BatchNorm/FusedBatchNorm")


		# uff wrong!!!
		parser.parse(model_file, network)

	# error Segmentation fault (core dumped) -> parser location
	#[TensorRT] ERROR: UFFParser: Validator error: TRTEngineOp_1: Unsupported operation _TRTEngineOp


	
	'''
	max_batch_size =1

	builder.max_batch_size = max_batch_size

	builder.max_workspace_size = (1 << 20)
	# This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.

	#with trt.Builder(TRT_LOGGER) as builder:

	with builder.build_cuda_engine(network) as engine:
		# Do inference here.

		print('test')

		serialized_engine = engine.serialize()
	'''





























if __name__ == '__main__':
    main()


    '''
    # Build a TensorRT engine.
    with build_engine_uff(uff_model_file) as engine:

        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.

        #h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)


        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:

            print('test!!!!!!!!!!!!!!!')

            serialized_engine = engine.serialize()


        # runtime
        #with trt.Runtime(TRT_LOGGER) as runtime:
        #	engine = runtime.deserialize_cuda_engine(serialized_engine)



        # Serialize the engine
    with open("./cpn_trt_FP16", "wb") as f:
        f.write(engine.serialize())
    '''



    '''
    # Load a normalized test case into the host input page-locked buffer.
    test_image = random.choice(test_images)
    test_case = load_normalized_test_case(test_image, h_input)
    # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
    # probability that the image corresponds to that label
    do_inference(context, h_input, d_input, h_output, d_output, stream)
    # We use the highest probability as our prediction. Its index corresponds to the predicted label.
    pred = labels[np.argmax(h_output)]
    if "_".join(pred.split()) in os.path.splitext(os.path.basename(test_case))[0]:
        print("Correctly recognized " + test_case + " as " + pred)
    else:
        print("Incorrectly recognized " + test_case + " as " + pred)
    '''






'''
if __name__ == '__main__':

	#with tf.device('/gpu:1'):


	# why run? "/usr/src/tensorrt/samples/python/introductory_parser_samples"

	model_file = "/host_temp/pose-estimation-trt/data/CPN_TENSORRT_FP16.uff"


	with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
		parser.register_input(cfg.INPUT_NODE, (32, 256, 192, 3))
		parser.register_output(cfg.OUTPUT_NODE)

	parser.parse(model_file, network)


	max_batch_size =1

	builder.max_batch_size = max_batch_size

	builder.max_workspace_size = 1 <<  20 
	# This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.

	#with trt.Builder(TRT_LOGGER) as builder:

	with builder.build_cuda_engine(network) as engine:
		# Do inference here.

		print('test')

		serialized_engine = engine.serialize()


	# runtime
	with trt.Runtime(TRT_LOGGER) as runtime:
		engine = runtime.deserialize_cuda_engine(serialized_engine)



	# Serialize the engine
	with open("./cpn_trt_FP16", "wb") as f:
			f.write(engine.serialize())


	# Deserialize the engine
	
	with open("./cpn_trt_FP16", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
			engine = runtime.deserialize_cuda_engine(f.read())
	
'''
