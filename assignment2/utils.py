import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from nst_utils import load_vgg_model

tf.compat.v1.disable_eager_execution()


class GramLossComputer:
    def __init__(self, content_reference: np.ndarray, style_reference: np.ndarray,
                 model="pretrained-model/imagenet-vgg-verydeep-19.mat"):
        model = load_vgg_model(model)
        content_reference = content_reference[np.newaxis, ...]
        self.sess = tf.compat.v1.InteractiveSession()
        self.sess.run(model['input'].assign(content_reference))
        out = model['conv4_2']
        a_C = self.sess.run(out)
        a_G = out
        self.J_content = self.compute_content_cost(a_C, a_G)
        style_reference = style_reference[np.newaxis, ...]
        self.sess.run(model['input'].assign(style_reference))
        self.STYLE_LAYERS = [
            ('conv2_1', 0.2),
            ('conv3_1', 0.4),
            ('conv4_1', 0.4),
        ]
        self.model = model
        self.J_style = self.compute_style_cost(model, self.STYLE_LAYERS)
        self.J = self.total_cost(self.J_content, self.J_style, alpha=10, beta=40)

    def gram_matrix(self, A):
        return tf.matmul(A, tf.transpose(A))

    def compute_content_cost(self, a_C, a_G):
        m, n_H, n_W, n_C = a_G.get_shape().as_list()

        a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
        a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

        J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
        return J_content

    def compute_layer_style_cost(self, a_S, a_G):
        m, n_H, n_W, n_C = a_G.get_shape().as_list()
        a_S = tf.transpose(tf.reshape(a_S, ([n_H * n_W, n_C])))
        a_G = tf.transpose(tf.reshape(a_G, ([n_H * n_W, n_C])))

        GS = self.gram_matrix(a_S)
        GG = self.gram_matrix(a_G)

        J_style_layer = 1. / (4 * n_C ** 2 * (n_H * n_W) ** 2) * tf.reduce_sum(tf.pow((GS - GG), 2))
        return J_style_layer

    def compute_style_cost(self, model, STYLE_LAYERS):
        J_style = 0
        for layer_name, coeff in STYLE_LAYERS:
            out = self.model[layer_name]
            a_S = self.sess.run(out)
            a_G = out
            J_style_layer = self.compute_layer_style_cost(a_S, a_G)
            J_style += coeff * J_style_layer
        return J_style

    def total_cost(self, J_content, J_style, alpha=10, beta=40):
        J = alpha * J_content + beta * J_style
        return J

    def fitness(self, picture):
        self.sess.run(self.model["input"].assign(picture[np.newaxis, ...]))
        return 1 / self.sess.run(self.J)