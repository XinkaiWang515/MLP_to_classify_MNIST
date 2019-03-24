import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
import time

def activation(kind,x):
    if kind=='relu' :
        return x * (x > 0)
    elif kind=='tanh':
        return np.tanh(x)

def softmax(x):
    numer = np.exp(x)
    deno = sum(numer)
    return numer / deno

def deriv(kind,x):
    if kind=='relu':
        return 1.0 * (x > 0)
    elif kind=='tanh':
        return 1.0 - np.tanh(x) ** 2

DATA_FNAME = 'mnist_traindata.hdf5'
test_data = 'mnist_testdata.hdf5'
save_file = 'xinkai_hw3p2.hdf5'

# f = h5py.File(DATA_FNAME,'r')
# f.visit(print)

with h5py.File(DATA_FNAME,'r') as hf:
    xdata = hf['xdata'][:]
    ydata = hf['ydata'][:]

with h5py.File(test_data,'r') as hf:
    x_test = hf['xdata'][:]
    y_test = hf['ydata'][:]

# epoch = 50
# minibatch = 500
def MLP_backprop_training(actF_category,eta,epoch,minibatch):
    x_train = xdata[:50000]
    x_val = xdata[50000:]
    y_train = ydata[:50000]
    y_val = ydata[50000:]

    W1 = 0.01 * np.random.randn(200, 784)
    W2 = 0.01 * np.random.randn(100, 200)
    W3 = 0.01 * np.random.randn(10, 100)
    b1 = 0.01 * np.random.randn(200)
    b2 = 0.01 * np.random.randn(100)
    b3 = 0.01 * np.random.randn(10)

    accu_train = []
    accu_val = []

    for i in range(epoch):
        if i==20:
            eta /= 2
        if i==40:
            eta /= 2

        for n in range(int(x_train.shape[0]/minibatch)):

            # delta3 = np.zeros((minibatch,W3.shape[0]))
            # delta2 = np.zeros((minibatch,W2.shape[0]))
            # delta1 = np.zeros((minibatch,W1.shape[0]))
            # a1 = np.zeros_like(delta1)
            # a2 = np.zeros_like(delta2)
            #
            # for idx,img in enumerate(x_train[n*minibatch:(n+1)*minibatch]):
            #     S1 = W1 @ img + b1
            #     a1[idx] = ReLU(S1)
            #     S2 = W2 @ a1[idx] + b2
            #     a2[idx] = ReLU(S2)
            #     S3 = W3 @ a2[idx] + b3
            #     out = softmax(S3)
            #
            #     der_a1 = ReLU_derivative(S1)
            #     der_a2 = ReLU_derivative(S2)
            #
            #     # C = -(y_train[n*minibatch+idx] @ np.log(out))
            #     delta3[idx] = out - y_train[n*minibatch+idx]
            #     delta2[idx] = (np.transpose(W3)@delta3[idx])*der_a2
            #     delta1[idx] = (np.transpose(W2)@delta2[idx])*der_a1

            out = np.zeros((minibatch,W3.shape[0]))
            S1 = x_train[n*minibatch:(n+1)*minibatch]@W1.T + np.tile(b1,(minibatch,1))
            a1 = activation(actF_category,S1)
            S2 = a1 @ W2.T + np.tile(b2, (minibatch, 1))
            a2 = activation(actF_category,S2)
            S3 = a2 @ W3.T + np.tile(b3, (minibatch, 1))
            for i in range(minibatch):
                out[i] = softmax(S3[i])
            der_a1 = deriv(actF_category,S1)
            der_a2 = deriv(actF_category,S2)

            # print(np.amax(out))
            delta3 = out - y_train[n*minibatch:(n+1)*minibatch]
            delta2 = (delta3@W3)*der_a2
            delta1 = (delta2@W2)*der_a1

            W3 -= eta*(delta3.T@a2)/minibatch
            W2 -= eta*(delta2.T@a1)/minibatch
            W1 -= eta*(delta1.T@x_train[n*minibatch:(n+1)*minibatch])/minibatch
            b3 -= eta*np.mean(delta3,axis=0)
            b2 -= eta*np.mean(delta2,axis=0)
            b1 -= eta*np.mean(delta1,axis=0)

        S = activation(actF_category,(activation(actF_category,(x_train@W1.T + np.tile(b1,(y_train.shape[0],1))))@W2.T + np.tile(b2,(y_train.shape[0],1))))@W3.T + np.tile(b3,(y_train.shape[0],1))
        tr_out = np.zeros_like(y_train)
        for i in range(x_train.shape[0]):
            tr_out[i] = softmax(S[i])
        tr_correct = np.sum(np.where(np.argmax(tr_out,axis=1)==np.argmax(y_train,axis=1),1,0))
        accu_train.append(tr_correct/y_train.shape[0])

        v_S = activation(actF_category,(activation(actF_category,x_val@W1.T + np.tile(b1,(y_val.shape[0],1)))@W2.T + np.tile(b2,(y_val.shape[0],1))))@W3.T + np.tile(b3,(y_val.shape[0],1))
        v_out = np.zeros_like(y_val)
        for i in range(y_val.shape[0]):
            v_out[i] = softmax(v_S[i])
        v_correct = np.sum(np.where(np.argmax(v_out, axis=1) == np.argmax(y_val, axis=1), 1, 0))
        accu_val.append(v_correct / y_val.shape[0])

    return np.array(accu_train),np.array(accu_val),epoch,minibatch


if __name__=='__main__':
    max_accu = 0
    for actFCN in ['relu','tanh']:
        for eta in [0.02,0.1,0.5]:
            start = time.time()
            accu_train,accu_val,epoch,minibatch = MLP_backprop_training(actFCN,eta,50,500)
            if accu_val[-1]>max_accu:
                max_accu = accu_val[-1]
                actF = actFCN
                lr = eta
            end = time.time()
            print('time per epoch', end - start)
            plt.title('activation function: %s,learning rate: %.2f'%(actFCN,eta))
            plt.plot(accu_train,label='training')
            plt.plot(accu_val,label='validation')
            plt.axvline(x=20)
            plt.axvline(x=40)
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            plt.legend()
            plt.show()

    W1 = 0.01 * np.random.randn(200, 784)
    W2 = 0.01 * np.random.randn(100, 200)
    W3 = 0.01 * np.random.randn(10, 100)
    b1 = 0.01 * np.random.randn(200)
    b2 = 0.01 * np.random.randn(100)
    b3 = 0.01 * np.random.randn(10)

    choose_lr = lr
    for i in range(epoch):
        if i == 20:
            lr /= 2
        if i == 40:
            lr /= 2

        for n in range(int(xdata.shape[0] / minibatch)):
            out = np.zeros((minibatch, W3.shape[0]))
            S1 = xdata[n * minibatch:(n + 1) * minibatch] @ W1.T + np.tile(b1, (minibatch, 1))
            a1 = activation(actF, S1)
            S2 = a1 @ W2.T + np.tile(b2, (minibatch, 1))
            a2 = activation(actF, S2)
            S3 = a2 @ W3.T + np.tile(b3, (minibatch, 1))
            for i in range(minibatch):
                out[i] = softmax(S3[i])
            der_a1 = deriv(actF, S1)
            der_a2 = deriv(actF, S2)

            # print(np.amax(out))
            delta3 = out - ydata[n * minibatch:(n + 1) * minibatch]
            delta2 = (delta3 @ W3) * der_a2
            delta1 = (delta2 @ W2) * der_a1

            W3 -= lr * (delta3.T @ a2) / minibatch
            W2 -= lr * (delta2.T @ a1) / minibatch
            W1 -= lr * (delta1.T @ xdata[n * minibatch:(n + 1) * minibatch]) / minibatch
            b3 -= lr * np.mean(delta3, axis=0)
            b2 -= lr * np.mean(delta2, axis=0)
            b1 -= lr * np.mean(delta1, axis=0)

    with h5py.File(save_file, 'w') as outfile:
        outfile.attrs['act'] = np.string_(actF)
        outfile.create_dataset('b1', data=b1)
        outfile.create_dataset('b2', data=b2)
        outfile.create_dataset('b3', data=b3)
        outfile.create_dataset('w1', data=W1)
        outfile.create_dataset('w2', data=W2)
        outfile.create_dataset('w3', data=W3)

    test_S = activation(actF, (
                activation(actF, x_test @ W1.T + np.tile(b1, (y_test.shape[0], 1))) @ W2.T + np.tile(b2, (
        y_test.shape[0], 1)))) @ W3.T + np.tile(b3, (y_test.shape[0], 1))
    y_predict = np.zeros_like(y_test)
    for i in range(y_test.shape[0]):
        y_predict[i] = softmax(test_S[i])
    test_correct = np.sum(np.where(np.argmax(y_predict, axis=1) == np.argmax(y_test, axis=1), 1, 0))
    te_accu = test_correct / y_test.shape[0]
    print('activation fuction: %s, learning rate: %.2f, test_accuracy: %.3f'%(actF,choose_lr,te_accu))




