import numpy as np
import tensorflow as tf
import cv2

class FeatureExtractor:
    def __init__(self, model_filename, input_shape=(128, 64)):
        self.input_shape = input_shape
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_filename, 'rb') as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.input_var = self.graph.get_tensor_by_name("images:0")
        self.output_var = self.graph.get_tensor_by_name("features:0")

    def __call__(self, images):
        preprocessed = np.asarray([
            cv2.resize(im, self.input_shape).astype(np.uint8) for im in images
        ])
        return self.sess.run(self.output_var, feed_dict={self.input_var: preprocessed})
