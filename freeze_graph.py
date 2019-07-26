import tensorflow as tf
from tensorflow.python.tools import freeze_graph

import os

def write_graph(graph_name, graph_def, output_dir):
	output_path = os.path.join(output_dir, graph_name)
	with tf.gfile.GFile(output_path, "wb") as f:
		f.write(graph_def.SerializeToString())

def get_frozen_graph(graph_file):

	with tf.gfile.FastGFile(graph_file, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def

def main():

	freeze_graph.freeze_graph('/graph.pbtxt', "", False, 
				'/checkpoint.ckpt', 'tower_0/refine_out/BatchNorm/FusedBatchNorm',
				"save/restore_all", "save/Const",
				'frozen.pb', True, "")

	print('done.')

	
if __name__ == '__main__':
	main()
