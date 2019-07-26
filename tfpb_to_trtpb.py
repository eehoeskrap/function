import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import tensorflow.contrib.tensorrt as trt
import os


def write_graph_to_file(graph_name, graph_def, output_dir):

	# Write Frozen Graph file to disk
	output_path = os.path.join(output_dir, graph_name)
	with tf.gfile.GFile(output_path, "wb") as f:
		f.write(graph_def.SerializeToString())

def get_frozen_graph(graph_file):

	#Read Frozen Graph file from disk
	with tf.gfile.FastGFile(graph_file, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	return graph_def


def main():

	# Open frozen.pb
	frozen_graph_def = get_frozen_graph('./TENSORFLOW_FROZEN.pb')


	# output node
	output_nodes = ['tower_0/refine_out/BatchNorm/FusedBatchNorm']


	# TensorRT inference graph
	trt_graph = trt.create_inference_graph(
		frozen_graph_def,
		output_nodes,
		max_batch_size=1,
		max_workspace_size_bytes=(2 << 10) << 20,
		precision_mode='FP16')

	print('!!!!!! trt graph create !!!!!!')

	
	# Write 'TRT_FP16.pb'
	write_graph_to_file('TRT_FP16.pb', trt_graph ,'./')
	
	# check how many ops of the original frozen model
	all_nodes = len([1 for n in frozen_graph_def.node])
	print("numb. of all_nodes in frozen graph:", all_nodes)

	# check how many ops that is converted to TensorRT engine
	trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
	print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)

	all_nodes = len([1 for n in trt_graph.node])
	print("numb. of all_nodes in TensorRT graph:", all_nodes)
	

if __name__ == '__main__':

    main()
