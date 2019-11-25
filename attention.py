import tensorflow as tf
from tensorflow.contrib import layers

class Attent:


    def __init__(self, textFinal, imgFinal, hidden_size1=75,  batch_size1=32):

        self.textFinal = textFinal
        self.imgFinal = imgFinal
        self.hidden_size = hidden_size1
        self.output_attention, self.inputs_text, self.inputs, self.text_all_tensor \
            = self.AttentionLayer(self.textFinal, self.imgFinal, 'Atten-Ins')

        self.output = self.FC(self.output_attention, 'FC')

    def FC(self, inputs, name):
        output = layers.fully_connected(inputs,self.hidden_size*2)
        return output


    def AttentionLayer(self, inputs, input_img, name):
        with tf.variable_scope(name):
            # c = np.zeros((self.batch_size, 200, 2))
            r_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='r_context')
            h1 = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)  # 文本
            h2 = layers.fully_connected(input_img, self.hidden_size*2, activation_fn=tf.nn.tanh)  # 图片 (2, 200, 150)
            sum_h1 = tf.reduce_sum(tf.multiply(h1, r_context), axis=2, keep_dims=True)  # (2,200,1)
            sum_h2 = tf.reduce_sum(tf.multiply(h2, r_context), axis=2, keep_dims=True)  # (2,200,1)
    
            c =tf.concat([sum_h1, sum_h2], 2)  # 2,200,2
            alpha = tf.nn.softmax(c, dim=2)  # 2,200,2
            text_all_tensor =alpha[:,:,0]
            img_all_tensor = alpha[:,:,1]
   
            inputs_text = inputs*text_all_tensor[:, :, None]
            inputs_img = h2*img_all_tensor[:, :, None]
            output_attention = tf.add(inputs_text, inputs_img)
        return output_attention, inputs_text, inputs, text_all_tensor