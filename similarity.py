import tensorflow as tf
from tensorflow.contrib import layers
from lstm import LSTM
from att_lstm import Att_LSTM
from attention import Attent
class similarity:

    def __init__(self, batch_size1=32):
        self.hidden_size1 = 75
        self.batch_size1 = 32  # 64
        self.embedding_size = 1400
        self.embedding_img = 2048
        self.lstm = LSTM(self.embedding_size, self.hidden_size1)  # 初始化
        self.att_lstm = Att_LSTM(self.embedding_size, self.embedding_img)
        self.input_tw_time = tf.placeholder(tf.float32, [batch_size1, None], name='input_tw_time')
        self.input_ins_time = tf.placeholder(tf.float32, [batch_size1, None], name='input_ins_time')
        self.input_y = tf.placeholder(tf.float32,[batch_size1], name='input_y')

        # Handle Time
        self.instime0 = tf.expand_dims(self.input_ins_time,2)
        self.instime = tf.tile(self.instime0, [1,1, 150])

        self.twtime0 = tf.expand_dims(self.input_tw_time,1)
        self.twtime = tf.tile(self.twtime0, [1,150,1])
        ##alc Proc
        self.timeres0 = self.instime-self.twtime
        self.timeres1 = tf.abs(self.timeres0)
        self.timeres2 = tf.log(self.timeres1)
        self.timeRes3 = tf.reciprocal(self.timeres2+1e-18) # <tf.Tensor 'Reciprocal:0' shape=(64, 150, 150) dtype=float32>
        self.Wt = tf.Variable(tf.ones([1,150,150]),dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([1,150,150]) + 0.0001,dtype=tf.float32)
        self.timeRes4 = self.timeRes3+1
        self.timeRes5 = tf.log(self.timeRes4)
        self.timeRes = self.Wt*self.timeRes5 + self.b


        # Handle Cosine


        self.twdata0 = self.lstm.output
        self.insdata0 = self.att_lstm.output
        self.insimg0 = self.att_lstm.insimg

        self.twdata = self.twdata0
        self.insdata = self.insdata0
        self.twTr = tf.transpose(self.twdata,[0,2,1])
        self.res1 = tf.matmul(self.insdata,self.twTr)
        self.norm1 = tf.ones([32,150,150],dtype=tf.float32)
        self.insNorm = tf.norm(self.insdata,axis=2)
        self.insNorm2 = tf.expand_dims(self.insNorm,2)
        self.twNorm = tf.norm(self.twTr,axis=1)
        self.twNorm2 = tf.expand_dims(self.twNorm,1)
        self.norm2 = self.norm1*self.insNorm2
        self.norm3 = self.norm2*self.twNorm2
        self.res2 = self.res1/self.norm3
        self.cosres = self.timeRes * self.res2



        self.twdataI = self.twdata0
        self.insdataI = self.insimg0
        self.twTrI = tf.transpose(self.twdataI,[0,2,1])
        self.res1I = tf.matmul(self.insdataI,self.twTrI)
        self.norm1I = tf.ones([32,150,150],dtype=tf.float32)
        self.insNormI = tf.norm(self.insdataI,axis=2)
        self.insNorm2I = tf.expand_dims(self.insNormI,2)
        self.twNormI = tf.norm(self.twTrI,axis=1)
        self.twNorm2I = tf.expand_dims(self.twNormI,1)
        self.norm2I = self.norm1I*self.insNorm2I
        self.norm3I = self.norm2I*self.twNorm2I
        self.res2I = self.res1I/self.norm3I
        self.cosresI = self.timeRes * self.res2I


##############################################


        self.Att = Attent(self.cosres,self.cosresI)

        self.res = self.Att.output
        self.simi = tf.reduce_mean(self.res,axis=2)

        self.Wsimi = tf.Variable(tf.truncated_normal([150, 2]))
        self.biases = tf.Variable(tf.zeros([1, 2]) + 0.00001)

        # self.y = 1.0 / (1.0 + tf.exp( -1.0*tf.matmul(self.simi, self.Wsimi) + self.biases))
        # self.loss = tf.reduce_mean(- self.input_y.reshape(-1, 1) *  tf.log(self.y) - (1 - self.input_y.reshape(-1, 1)) * tf.log(1 - self.y))


        self.outputs = tf.matmul(self.simi, self.Wsimi) + self.biases
        self.predictions = tf.nn.softmax(self.outputs)
        self.y_label = tf.cast(self.input_y, tf.int32)
        self.ys = tf.one_hot(self.y_label,2,1,0)
        self.y_label2 = tf.cast(self.ys, tf.float32)
        
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.y_label2)
        self.loss = tf.reduce_mean(self.cross_entropy)
        self.train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

        self.correct_predictions = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.ys, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

