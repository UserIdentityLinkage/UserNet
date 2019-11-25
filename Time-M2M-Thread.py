#!/usr/bin/env python
# coding: utf-8

import os
import datetime
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

import numpy as np
import tensorflow as tf
from similarity import similarity
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor , as_completed
from sklearn.metrics import accuracy_score 
if not os.path.exists("./log/"):
    os.mkdir("./log/")

baseDir = "./data/"
batch_size1 = 32 # 64
embedding_size = 1400
embedding_img = 2048
length = 256
allEpoch = 1000

saveModel = True
loadModel = False
TestOnly = False

TrainLoadData = 6
AllTrainLen = 36#36
TrainSt = 0
LoadTrainTime = int(AllTrainLen/TrainLoadData)

ValidLoadData = 4
AllValidLen = 4#4
ValidSt = 0
LoadValidTime = int(AllValidLen/ValidLoadData)


TestLoadData = 4
AllTestLen = 4#4
TestSt = 0
LoadTestTime = int(AllTestLen/TestLoadData)

# Train
arr_tw_textAll=[]
arr_tw_timeAll=[]
arr_ins_textAll=[]
arr_ins_imgAll=[]
arr_ins_timeAll=[]
arr_labelAll=[]

#Test
arr_tw_textTest = []
arr_tw_timeTest = []
arr_ins_textTest = []
arr_ins_imgTest = []
arr_ins_timeTest = []
arr_labelTest = []

#Valid
arr_tw_textValid = []
arr_tw_timeValid = []
arr_ins_textValid = []
arr_ins_imgValid = []
arr_ins_timeValid = []
arr_labelValid = []

def loadTrainThread(st):
    ti = str(st)
    path_tw_text = baseDir+r'Traindata/twTrainText'+ti+'.npy'
    path_tw_time = baseDir+r'Traindata/twTrainTime'+ti+'.npy'
    path_ins_text = baseDir+r'Traindata/insTrainText'+ti+'.npy'
    path_ins_img = baseDir+r'Traindata/insTrainImg'+ti+'.npy'
    path_ins_time = baseDir+r'Traindata/insTrainTime'+ti+'.npy'
    path_label = baseDir+r'Traindata/yTrain'+ti+'.npy'

    arr_tw_text = np.load(path_tw_text)

    arr_tw_time = np.load(path_tw_time)

    arr_ins_text = np.load(path_ins_text)

    arr_ins_img = np.load(path_ins_img)

    arr_ins_time = np.load(path_ins_time)

    arr_label = np.load(path_label)
    return arr_tw_text,arr_tw_time,arr_ins_text,arr_ins_img,arr_ins_time,arr_label


# In[6]:


def clearTrain():
    global arr_tw_textAll,arr_tw_timeAll,arr_ins_textAll,arr_ins_imgAll,arr_ins_timeAll,arr_labelAll

    arr_tw_textAll=[]
    arr_tw_timeAll=[]
    arr_ins_textAll=[]
    arr_ins_imgAll=[]
    arr_ins_timeAll=[]
    arr_labelAll=[]


# In[7]:



def loadTrain():
    global TrainSt
    print("loading Train data ",TrainSt," ",end="")
    clearTrain()
    executor=ThreadPoolExecutor()
    starttime = datetime.datetime.now()
    task = []
    for iii in range(TrainSt,TrainSt+TrainLoadData):
        curId = (iii*256)
        task.append(executor.submit(loadTrainThread,curId))
#     print("Submitted")
    TrainSt = (TrainSt+TrainLoadData)%AllTrainLen
    for iii in range(len(task)):
        arr_tw_text,arr_tw_time,arr_ins_text,arr_ins_img,arr_ins_time,arr_label = task[iii].result()
        arr_tw_textAll.append(arr_tw_text)
        arr_tw_timeAll.append(arr_tw_time)
        arr_ins_textAll.append(arr_ins_text)
        arr_ins_imgAll.append(arr_ins_img)
        arr_ins_timeAll.append(arr_ins_time)
        arr_labelAll.append(arr_label)

    executor.shutdown()
    # execu = ProcessPoolExecutor(max_workers=2)
    endtime = datetime.datetime.now()
    print("Time Cost: ",(endtime - starttime))


# In[8]:


def clearTest():
    global arr_tw_textTest,arr_tw_timeTest,arr_ins_textTest,arr_ins_imgTest,arr_ins_timeTest,arr_labelTest
    arr_tw_textTest = []
    arr_tw_timeTest = []
    arr_ins_textTest = []
    arr_ins_imgTest = []
    arr_ins_timeTest = []
    arr_labelTest = []


# In[9]:


def loadTestThread(st):
    ti = str(st)    
    path_tw_text = baseDir+r'Testdata/twTestText'+ti+'.npy'
    path_tw_time = baseDir+r'Testdata/twTestTime'+ti+'.npy'
    path_ins_text = baseDir+r'Testdata/insTestText'+ti+'.npy'
    path_ins_img = baseDir+r'Testdata/insTestImg'+ti+'.npy'
    path_ins_time = baseDir+r'Testdata/insTestTime'+ti+'.npy'
    path_label = baseDir+r'Testdata/yTest'+ti+'.npy'

    arr_tw_text = np.load(path_tw_text)
    
    arr_tw_time = np.load(path_tw_time)
    
    arr_ins_text = np.load(path_ins_text)
    
    arr_ins_img = np.load(path_ins_img)
    
    arr_ins_time = np.load(path_ins_time)
    
    arr_label = np.load(path_label)
    return arr_tw_text,arr_tw_time,arr_ins_text,arr_ins_img,arr_ins_time,arr_label


def loadTest():
    global TestSt
    print("loading Test data...",TestSt," ", end="")
    clearTest()
    starttime = datetime.datetime.now()
    executor=ThreadPoolExecutor()
    starttime = datetime.datetime.now()
    task = []
    for iii in range(TestSt,TestSt+TestLoadData):
        curId = (iii*256)
        task.append(executor.submit(loadTestThread,curId))
    TestSt = (TestSt+TestLoadData)%AllTestLen
    for iii in range(len(task)):
        arr_tw_text,arr_tw_time,arr_ins_text,arr_ins_img,arr_ins_time,arr_label = task[iii].result()
        arr_tw_textTest.append(arr_tw_text)
        arr_tw_timeTest.append(arr_tw_time)
        arr_ins_textTest.append(arr_ins_text)
        arr_ins_imgTest.append(arr_ins_img)
        arr_ins_timeTest.append(arr_ins_time)
        arr_labelTest.append(arr_label)



    executor.shutdown()
    endtime = datetime.datetime.now()
    print(" Time Cost: ",(endtime - starttime))
    
def clearValid():
    global arr_tw_textValid,arr_tw_timeValid,arr_ins_textValid,arr_ins_imgValid,arr_ins_timeValid,arr_labelValid
    arr_tw_textValid = []
    arr_tw_timeValid = []
    arr_ins_textValid = []
    arr_ins_imgValid = []
    arr_ins_timeValid = []
    arr_labelValid = []

def loadValidThread(st):
    ti = str(st)    
    path_tw_text = baseDir+r'Validdata/twValidText'+ti+'.npy'
    path_tw_time = baseDir+r'Validdata/twValidTime'+ti+'.npy'
    path_ins_text = baseDir+r'Validdata/insValidText'+ti+'.npy'
    path_ins_img = baseDir+r'Validdata/insValidImg'+ti+'.npy'
    path_ins_time = baseDir+r'Validdata/insValidTime'+ti+'.npy'
    path_label = baseDir+r'Validdata/yValid'+ti+'.npy'

    arr_tw_text = np.load(path_tw_text)
    
    arr_tw_time = np.load(path_tw_time)
    
    arr_ins_text = np.load(path_ins_text)
    
    arr_ins_img = np.load(path_ins_img)
    
    arr_ins_time = np.load(path_ins_time)
    
    arr_label = np.load(path_label)
    return arr_tw_text,arr_tw_time,arr_ins_text,arr_ins_img,arr_ins_time,arr_label
def loadValid():
    global ValidSt
    print("loading Valid data...",ValidSt," ", end="")
    clearValid()
    starttime = datetime.datetime.now()
    executor=ThreadPoolExecutor()
    task = []
    for iii in range(ValidSt,ValidSt+ValidLoadData):
        curId = (iii*256)
        task.append(executor.submit(loadValidThread,curId))
    ValidSt = (ValidSt+ValidLoadData)%AllValidLen
    for iii in range(len(task)):
        arr_tw_text,arr_tw_time,arr_ins_text,arr_ins_img,arr_ins_time,arr_label = task[iii].result()
        arr_tw_textValid.append(arr_tw_text)
        arr_tw_timeValid.append(arr_tw_time)
        arr_ins_textValid.append(arr_ins_text)
        arr_ins_imgValid.append(arr_ins_img)
        arr_ins_timeValid.append(arr_ins_time)
        arr_labelValid.append(arr_label)
    executor.shutdown()
    endtime = datetime.datetime.now()
    print(" Time Cost: ",(endtime - starttime))

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.15
# tf.reset_default_graph()
sess = tf.Session(config=config)
print("Loading Model",end="")
similarityV = similarity(batch_size1)
saver = tf.train.Saver(max_to_keep=3)
sess.run(tf.global_variables_initializer())
if loadModel:
    model_file=tf.train.latest_checkpoint('./model/')
    saver.restore(sess,model_file)

print(" finally", end="")

print(" finished")

def npyCos(vector1,vector2):
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

def timeFactor(twVec,insVec,x_tw_time_batch,x_ins_time_batch):
    resVec = []
    for bat in range(batch_size1):
        curVec = []
        for itm in range(150):
            cosSum = 0.0
            for itm2 in range(150):
                cosdata = npyCos(twVec[bat][itm],insVec[bat][itm2])
                twTi = x_tw_time_batch[bat][itm]
                insTi = x_ins_time_batch[bat][itm2]

                TimeC = 1.0/(math.log(float(abs(insTi-twTi))+1e-18)+1e-18)
                curCoe = cosdata*TimeC
                cosSum = cosSum+curCoe
            curVec.append(cosSum)
        resVec.append(curVec)
    return resVec

def shuffleData(a,b,c,d,e,f):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state) 
    np.random.shuffle(b)
    np.random.set_state(state) 
    np.random.shuffle(c)
    np.random.set_state(state) 
    np.random.shuffle(d)
    np.random.set_state(state) 
    np.random.shuffle(e)
    np.random.set_state(state) 
    np.random.shuffle(f)
if not TestOnly:
    logbase = "./log/"
    trainBatchAcc = open(logbase+"trainBatchAcc.txt","w")
    trainAllBatchAcc = open(logbase+"trainAllBatchAcc.txt","w")
    trainEpochAcc = open(logbase+"trainEpochAcc.txt","w")
    trainBatchLoss = open(logbase+"trainBatchLoss.txt","w")
    trainAllBatchLoss = open(logbase+"trainAllBatchLoss.txt","w")
    trainEpochLoss = open(logbase+"trainEpochLoss.txt","w")
    testAccW = open(logbase+"testAcc.txt","w")
    testLossW = open(logbase+"testLoss.txt","w")

    ValidAccW = open(logbase+"validAcc.txt","w")
    ValidLossW = open(logbase+"validLoss.txt","w")

if __name__ == "__main__":
    curEpoch = 0
    max_acc=0.0

    if TestOnly:
        ## Test
        TestAcc = 0.0
        TestLoss = 0.0
        for loadload in range(LoadTestTime):
            if AllTrainLen!=1:
                loadTest()
            startCalc = datetime.datetime.now()
            for cur in range(TestLoadData):
                arr_tw_text = arr_tw_textTest[cur]
                arr_tw_time = arr_tw_timeTest[cur]
                arr_ins_text = arr_ins_textTest[cur]
                arr_ins_img = arr_ins_imgTest[cur]
                arr_ins_time = arr_ins_timeTest[cur]
                arr_label = arr_labelTest[cur]
                tloss = 0
                acc = 0
                acc_every = 0
                loss_every = 0.0
                for i in range(0, length, batch_size1):   
                    y_label_batch = arr_label[i: i+batch_size1]
                    x_tw_text_batch = arr_tw_text[i: i + batch_size1]
                    x_tw_time_batch = arr_tw_time[i: i+batch_size1]
                    x_ins_text_batch = arr_ins_text[i: i + batch_size1]
                    x_ins_img_batch = arr_ins_img[i: i + batch_size1]
                    x_ins_time_batch = arr_ins_time[i: i + batch_size1]

                    cost, accuracy = sess.run(
                        [similarityV.loss, similarityV.accuracy],
                        feed_dict={similarityV.lstm.input_x: x_tw_text_batch,
                                similarityV.input_tw_time: x_tw_time_batch,
                                similarityV.att_lstm.input_x:x_ins_text_batch,
                                similarityV.input_ins_time:x_ins_time_batch,
                                similarityV.att_lstm.input_image:x_ins_img_batch,
                                similarityV.input_y: y_label_batch
                                })
                    
                    acc_every = acc_every+accuracy
                    loss_every = loss_every+cost
                acc = acc_every/8.0 # 256/64
                tloss = loss_every/8.0
                print('Batch Test Accuracy', acc," Loss ",tloss)
                TestAcc = TestAcc + acc
                TestLoss = TestLoss+tloss
            endCalc = datetime.datetime.now()
            print("Test Time = ",endCalc-startCalc)
            if AllTrainLen!=1:
                clearTest()

        TestAcc = TestAcc/(LoadTestTime*1.0*TestLoadData)
        TestLoss = TestLoss/(LoadTestTime*1.0*TestLoadData)
        print("Test Accuracy = ",TestAcc," Loss = ",TestLoss)
        exit()
    while curEpoch < allEpoch:
        curEpoch = curEpoch + 1
        BatchAcc = 0.0
        BatchLoss = 0.0
        EpochAcc = 0.0
        EpochLoss = 0.0
        # Train Batch
        startCalc = datetime.datetime.now()
        for loadload in range(LoadTrainTime):
            if AllTrainLen!=1:
                loadTrain()
            for cur in range(TrainLoadData):
                arr_tw_text = arr_tw_textAll[cur]
                arr_tw_time = arr_tw_timeAll[cur]
                arr_ins_text = arr_ins_textAll[cur]
                arr_ins_img = arr_ins_imgAll[cur]
                arr_ins_time = arr_ins_timeAll[cur]
                arr_label = arr_labelAll[cur]
                shuffleData(arr_tw_text,arr_tw_time,arr_ins_text,arr_ins_img,arr_ins_time,arr_label)
                acc = 0.0
                loss = 0.0
                acc_every = 0.0
                loss_every = 0.0
                for i in range(0, length, batch_size1):   
                    y_label_batch = arr_label[i: i+batch_size1]
                    x_tw_text_batch = arr_tw_text[i: i + batch_size1]
                    x_tw_time_batch = arr_tw_time[i: i+batch_size1]
                    x_ins_text_batch = arr_ins_text[i: i + batch_size1]
                    x_ins_img_batch = arr_ins_img[i: i + batch_size1]
                    x_ins_time_batch = arr_ins_time[i: i + batch_size1]
                    _,cost, accuracy= sess.run(
                        [similarityV.train_step,similarityV.loss, similarityV.accuracy],
                        feed_dict={similarityV.lstm.input_x: x_tw_text_batch,
                                similarityV.input_tw_time: x_tw_time_batch,
                                similarityV.att_lstm.input_x:x_ins_text_batch,
                                similarityV.input_ins_time:x_ins_time_batch,
                                similarityV.att_lstm.input_image:x_ins_img_batch,
                                similarityV.input_y: y_label_batch
                                })
                    acc_every = acc_every+accuracy
                    loss_every = loss_every+cost
                acc = acc_every/8.0 # 256/64
                loss = loss_every/8.0
                BatchLoss = BatchLoss+loss
                BatchAcc = BatchAcc + acc
                # print('Batch Epoch',curEpoch," - ",loadload," - ",cur, end="")
                # print(' Accuracy', acc," Loss ",cost)
                trainAllBatchAcc.write(str(acc))
                trainAllBatchAcc.write("\n")
                trainAllBatchAcc.flush()
                trainAllBatchLoss.write(str(loss))
                trainAllBatchLoss.write("\n")
                trainAllBatchLoss.flush()
        endCalc = datetime.datetime.now()
        print('All Batch Epoch',curEpoch, end="")
        BatchAcc = BatchAcc/(TrainLoadData*1.0*LoadTrainTime)
        BatchLoss = BatchLoss/(TrainLoadData*1.0*LoadTrainTime)
        print(' Accuracy', BatchAcc," Loss ",BatchLoss)

        trainBatchAcc.write(str(BatchAcc))
        trainBatchAcc.write("\n")
        trainBatchAcc.flush()
        trainBatchLoss.write(str(BatchLoss))
        trainBatchLoss.write("\n")
        trainBatchLoss.flush()

        print("Train Batch Time = ",endCalc-startCalc)
        if AllTrainLen!=1:
            clearTrain()

        # Train Epoch
        startCalc = datetime.datetime.now()
        for loadload in range(LoadTrainTime):
            if AllTrainLen!=1:
                loadTrain()
            for cur in range(TrainLoadData):
                arr_tw_text = arr_tw_textAll[cur]
                arr_tw_time = arr_tw_timeAll[cur]
                arr_ins_text = arr_ins_textAll[cur]
                arr_ins_img = arr_ins_imgAll[cur]
                arr_ins_time = arr_ins_timeAll[cur]
                arr_label = arr_labelAll[cur]

                acc = 0.0
                loss = 0.0
                acc_every = 0.0
                loss_every = 0.0
                for i in range(0, length, batch_size1):   
                    y_label_batch = arr_label[i: i+batch_size1]
                    x_tw_text_batch = arr_tw_text[i: i + batch_size1]
                    x_tw_time_batch = arr_tw_time[i: i+batch_size1]
                    x_ins_text_batch = arr_ins_text[i: i + batch_size1]
                    x_ins_img_batch = arr_ins_img[i: i + batch_size1]
                    x_ins_time_batch = arr_ins_time[i: i + batch_size1]

                    cost, accuracy = sess.run(
                        [similarityV.loss, similarityV.accuracy],
                        feed_dict={similarityV.lstm.input_x: x_tw_text_batch,
                                similarityV.input_tw_time: x_tw_time_batch,
                                similarityV.att_lstm.input_x:x_ins_text_batch,
                                similarityV.input_ins_time:x_ins_time_batch,
                                similarityV.att_lstm.input_image:x_ins_img_batch,
                                similarityV.input_y: y_label_batch
                                })
                    
                    acc_every = acc_every+accuracy
                    loss_every = loss_every+cost
                acc = acc_every/8.0  #((length*1.0)/batch_size1) # 256/64
                loss = loss_every/8.0  #((length*1.0)/batch_size1)
                EpochLoss = EpochLoss+loss
                EpochAcc = EpochAcc + acc
            if AllTrainLen!=1:
                clearTrain()
        endCalc = datetime.datetime.now()
        print('All Batch Epoch',curEpoch, end="")
        EpochLoss = EpochLoss/(TrainLoadData*1.0*LoadTrainTime)
        EpochAcc = EpochAcc/(TrainLoadData*1.0*LoadTrainTime)
        print(' Accuracy', EpochAcc," Loss ",EpochLoss)
        trainEpochAcc.write(str(EpochAcc))
        trainEpochAcc.write("\n")
        trainEpochAcc.flush()
        trainEpochLoss.write(str(EpochLoss))
        trainEpochLoss.write("\n")
        trainEpochLoss.flush()
        print("Train Batch Time = ",endCalc-startCalc)
        
        ## Valid
        ValidAcc = 0.0
        ValidLoss = 0.0
        for loadload in range(LoadValidTime):
            if AllTrainLen!=1:
                loadValid()
            startCalc = datetime.datetime.now()
            for cur in range(ValidLoadData):
                arr_tw_text = arr_tw_textValid[cur]
                arr_tw_time = arr_tw_timeValid[cur]
                arr_ins_text = arr_ins_textValid[cur]
                arr_ins_img = arr_ins_imgValid[cur]
                arr_ins_time = arr_ins_timeValid[cur]
                arr_label = arr_labelValid[cur]
                tloss = 0
                acc = 0
                acc_every = 0
                loss_every = 0.0
                for i in range(0, length, batch_size1):   
                    y_label_batch = arr_label[i: i+batch_size1]
                    x_tw_text_batch = arr_tw_text[i: i + batch_size1]
                    x_tw_time_batch = arr_tw_time[i: i+batch_size1]
                    x_ins_text_batch = arr_ins_text[i: i + batch_size1]
                    x_ins_img_batch = arr_ins_img[i: i + batch_size1]
                    x_ins_time_batch = arr_ins_time[i: i + batch_size1]

                    cost, accuracy = sess.run(
                        [similarityV.loss, similarityV.accuracy],
                        feed_dict={similarityV.lstm.input_x: x_tw_text_batch,
                                similarityV.input_tw_time: x_tw_time_batch,
                                similarityV.att_lstm.input_x:x_ins_text_batch,
                                similarityV.input_ins_time:x_ins_time_batch,
                                similarityV.att_lstm.input_image:x_ins_img_batch,
                                similarityV.input_y: y_label_batch
                                })
                    
                    acc_every = acc_every+accuracy
                    loss_every = loss_every+cost
                acc = acc_every/8.0 # 256/64
                tloss = loss_every/8.0
                print('Batch Valid Accuracy', acc," Loss ",tloss)
                ValidAcc = ValidAcc + acc
                ValidLoss = ValidLoss+tloss
            endCalc = datetime.datetime.now()
            print("Valid Time = ",endCalc-startCalc)
            if AllTrainLen!=1:
                clearValid()

        ValidAcc = ValidAcc/(LoadValidTime*1.0*ValidLoadData)
        ValidLoss = ValidLoss/(LoadValidTime*1.0*ValidLoadData)
        print("Valid Accuracy = ",ValidAcc," Loss = ",ValidLoss)
        if ValidAcc>max_acc:
            max_acc = ValidAcc
            if saveModel:
                saver.save(sess, './model/model.ckpt')
        ValidAccW.write(str(ValidAcc))
        ValidAccW.write("\n")
        ValidAccW.flush()
        ValidLossW.write(str(ValidLoss))
        ValidLossW.write("\n")
        ValidLossW.flush()

        ## Test
        TestAcc = 0.0
        TestLoss = 0.0
        for loadload in range(LoadTestTime):
            if AllTrainLen!=1:
                loadTest()
            startCalc = datetime.datetime.now()
            for cur in range(TestLoadData):
                arr_tw_text = arr_tw_textTest[cur]
                arr_tw_time = arr_tw_timeTest[cur]
                arr_ins_text = arr_ins_textTest[cur]
                arr_ins_img = arr_ins_imgTest[cur]
                arr_ins_time = arr_ins_timeTest[cur]
                arr_label = arr_labelTest[cur]
                tloss = 0
                acc = 0
                acc_every = 0
                loss_every = 0.0
                for i in range(0, length, batch_size1):   
                    y_label_batch = arr_label[i: i+batch_size1]
                    x_tw_text_batch = arr_tw_text[i: i + batch_size1]
                    x_tw_time_batch = arr_tw_time[i: i+batch_size1]
                    x_ins_text_batch = arr_ins_text[i: i + batch_size1]
                    x_ins_img_batch = arr_ins_img[i: i + batch_size1]
                    x_ins_time_batch = arr_ins_time[i: i + batch_size1]

                    cost, accuracy = sess.run(
                        [similarityV.loss, similarityV.accuracy],
                        feed_dict={similarityV.lstm.input_x: x_tw_text_batch,
                                similarityV.input_tw_time: x_tw_time_batch,
                                similarityV.att_lstm.input_x:x_ins_text_batch,
                                similarityV.input_ins_time:x_ins_time_batch,
                                similarityV.att_lstm.input_image:x_ins_img_batch,
                                similarityV.input_y: y_label_batch
                                })
                    
                    acc_every = acc_every+accuracy
                    loss_every = loss_every+cost
                acc = acc_every/8.0 # 256/64
                tloss = loss_every/8.0
                print('Batch Test Accuracy', acc," Loss ",tloss)
                TestAcc = TestAcc + acc
                TestLoss = TestLoss+tloss
            endCalc = datetime.datetime.now()
            print("Test Time = ",endCalc-startCalc)
            if AllTrainLen!=1:
                clearTest()

        TestAcc = TestAcc/(LoadTestTime*1.0*TestLoadData)
        TestLoss = TestLoss/(LoadTestTime*1.0*TestLoadData)
        print("Test Accuracy = ",TestAcc," Loss = ",TestLoss)
        # if TestAcc>max_acc:
        #     max_acc = TestAcc
        #     if saveModel:
        #         saver.save(sess, './model/model.ckpt')
        testAccW.write(str(TestAcc))
        testAccW.write("\n")
        testAccW.flush()
        testLossW.write(str(TestLoss))
        testLossW.write("\n")
        testLossW.flush()
    print("MAX Valid ACC = ",max_acc)
