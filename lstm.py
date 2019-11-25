import tensorflow as tf
import numpy as np

class LSTM:
    def __init__(self, embedding_size, hidden_size1=75,  batch_size1=32):
        self.input_x = tf.placeholder(tf.float32, [None, None, embedding_size], name='input_x')
        self.batch_size = batch_size1
        self.word_embeddings = embedding_size
        self.hidden_size = hidden_size1
        self.output = self.Bilstm(self.input_x, 'Twitter')

    def Bilstm(self, inputs, name):
        bi_output = []
        with tf.variable_scope(name):
            word_embedded = tf.reshape(inputs, [self.batch_size, -1, self.word_embeddings])
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=word_embedded, dtype=tf.float32)  # sequence_length为输入序列的实际长度
            output = tf.concat([output_fw, output_bw], axis=2)  # 0 (2,400,75)

 
        return output
