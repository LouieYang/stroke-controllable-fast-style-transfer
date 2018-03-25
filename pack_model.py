import os
import tensorflow as tf
from tensorflow import graph_util
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', required=True)
parser.add_argument('--output', dest='output', required=True)

args = parser.parse_args()

meta_graph = [meta for meta in os.listdir(args.checkpoint_dir) if '.meta' in meta]
assert (len(meta_graph) > 0)

sess = tf.Session()
saver = tf.train.import_meta_graph(os.path.join(args.checkpoint_dir, meta_graph[0]))
saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))
graph = tf.get_default_graph()

input_graph_def = graph.as_graph_def()

output_node_names = 'add_39'
output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def,output_node_names.split(","))

with tf.gfile.GFile(args.output, "wb") as f:
    f.write(output_graph_def.SerializeToString())
sess.close()