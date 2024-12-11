#!/usr/bin/env python
# coding: utf-8

# # DLVU - Assigment 1 - Ángel Amando Gil Álamo          

# ### Working out the fordward and backward pass (Questions 1-3)

# In[14]:


# Importing libraries
import math as mt
import numpy as np
import random as rd
import matplotlib.pyplot as plt

# Given data
x  = [1.,-1.]
w  = [[1.,1.,1.],[-1.,-1.,-1.]]
b1 = [0.,0.,0.]
v  = [[1.,1.],[-1.,-1.],[-1.,-1.]]
b2 = [0.,0.]
c  = 0

# Dimensions
dim_in  = 2 # Dimension of the Input layer
dim_hid  = 3 # Dimension of the first layer
dim_out = 2 # Dimension of the Output layer

# Derived data
k = np.zeros(dim_hid)
for j in range(dim_hid):
    for i in range(dim_in):
        k[j] += w[i][j] * x[i]
    k[j] += b1[j]

h = [1 / (1+mt.exp(-i)) for i in k]

o = np.zeros(dim_out)
for j in range(dim_out):
    for i in range(dim_hid):
        o[j] += v[i][j] * h[i]
    o[j] += b2[j]

softSum = 0.
for i in o:
    softSum+= mt.exp(i)
y = [mt.exp(i) / softSum for i in o]

l = 0.
for i in range(dim_out):
    l += -mt.log(y[c])

print('k=',k,'\nh=',h,'\no=',o,'\ny=',y,'\nl=',l)


# ### Backward pass

# In[6]:


y_d  = [-1/y[i] for i in range(dim_out)]
o_d  = [y[i] for i in range(dim_out)]
o_d[c]-=1

v_d = np.zeros((dim_hid,dim_out))
h_d  = np.zeros(dim_hid)
for j in range(dim_out):
    for i in range(dim_hid):
        v_d[i][j] += o_d[j] * h[i]
        h_d[i] += o_d[j] * v[i][j]       
b2_d = o_d
k_d  = [h_d[i]*h[i]*(1-h[i]) for i in range(dim_hid)]

w_d  = np.zeros((dim_in,dim_hid))
for j in range(dim_hid): 
    for i in range(dim_in): 
        w_d[i][j] = k_d[j] * x[i] 
b1_d = k_d

print('v_d=',v_d,'\nb2_d=',b2_d,'\nw_d=',w_d,'\nb1_d=',b1_d)


# ### Github code

# In[24]:


import numpy as np
from urllib import request
import gzip
import pickle
import os

def load_synth(num_train=60_000, num_val=10_000, seed=0):
    """
    Load some very basic synthetic data that should be easy to classify. Two features, so that we can 
    plot the decision boundary (which is an ellipse in the feature space).
    :param num_train: Number of training instances
    :param num_val: Number of test/validation instances
    :param num_features: Number of features per instance
    :return: Two tuples and an integer: (xtrain, ytrain), (xval, yval), num_cls. The first contains a 
    matrix of training data with 2 features as a numpy floating point array, and the corresponding 
    classification labels as a numpy integer array. The second contains the test/validation data in the 
    same format. The last integer contains the number of classes (this is always 2 for this function).
    """
    np.random.seed(seed)

    THRESHOLD = 0.6
    quad = np.asarray([[1, -0.05], [1, .4]])

    ntotal = num_train + num_val

    x = np.random.randn(ntotal, 2)

    # compute the quadratic form
    q = np.einsum('bf, fk, bk -> b', x, quad, x)
    y = (q > THRESHOLD).astype(np.int)

    return (x[:num_train, :], y[:num_train]), (x[num_train:, :], y[num_train:]), 2

def load_mnist(final=False, flatten=True):
    """
    Load the MNIST data.
    :param final: If true, return the canonical test/train split. If false, split some validation data 
    from the trainingdata and keep the test data hidden.
    :param flatten: If true, each instance is flattened into a vector, so that the data is returns as a 
    matrix with 768columns. If false, the data is returned as a 3-tensor preserving each image as a matrix.
    :return: Two tuples and an integer: (xtrain, ytrain), (xval, yval), num_cls. The first contains a 
    matrix of training data and the corresponding classification labels as a numpy integer array. The 
    second contains the test/validation data in the same format. The last integer contains the number of 
    classes (this is always 2 for this function).
     """

    if not os.path.isfile('mnist.pkl'):
        init()

    xtrain, ytrain, xtest, ytest = load()
    xtl, xsl = xtrain.shape[0], xtest.shape[0]

    if flatten:
        xtrain = xtrain.reshape(xtl, -1)
        xtest  = xtest.reshape(xsl, -1)

    if not final: # return the flattened images
        return (xtrain[:-5000], ytrain[:-5000]), (xtrain[-5000:], ytrain[-5000:]), 10

    return (xtrain, ytrain), (xtest, ytest), 10

# Numpy-only MNIST loader. Courtesy of Hyeonseok Jung
# https://github.com/hsjeong5/MNIST-for-Numpy

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


# ### Implementing a training loop (Question 4)
#     I left everything as basic as possible. There are only 2 extra things:
#     - The learning rate 'alpha' adapts on the fly with a simple formula: 
#           alpha=alpha*(actual/previous) 
#       where 'actual' is the mean loss of the last 100 iters and 'previous', the earlier 100.
#     - there are two vectors regarding loss: 
#           'loss': resets every epoch. It contains all loses of given epoch.
#           'loss_clustered': doesn't reset. It contains the average loss of every 100 iterations.    
#     The loss vectors are used for visualizing purposes. 

# In[16]:


# Importing libraries
import math as mt
import numpy as np
import random as rd
import matplotlib.pyplot as plt

# Loading the synthetic data
(xtrain, ytrain), (xval, yval), num_cls = load_synth()

# Dimensions of the neural network
dim_in  = 2
dim_hid  = 3
dim_out = 2

# Initialization
w  = np.random.normal(0,1,size=(dim_in,dim_hid))
b1 = np.zeros(dim_hid)
v  = np.random.normal(0,1,size=(dim_hid,dim_out))
b2 = np.zeros(dim_out)

alpha = 0.1
update_step = 100 # Used in the loop that updates alpha and the clustered loss
loss_clustered = []

epochs = 5
for epoch in range(epochs):
    loss = []
    for iter in range(len(xtrain)):
        x=xtrain[iter]
        c=ytrain[iter]
        # Implementation of the NN as in the previous exercises
    
        # Forward pass
        k = np.zeros(dim_hid)
        for j in range(dim_hid):
            for i in range(dim_in):
                k[j] += w[i][j] * x[i]
            k[j] += b1[j]

        h = [1 / (1+mt.exp(-i)) for i in k]

        o = np.zeros(dim_out)
        for j in range(dim_out):
            for i in range(dim_hid):
                o[j] += v[i][j] * h[i]
            o[j] += b2[j]

        softSum = 0.
        for i in o:
            softSum+= mt.exp(i)
        y = [mt.exp(i) / softSum for i in o]

        # Loss
        l = -mt.log(y[c])
        loss.append(l)
        
        # Alpha and clustered vector loop
        if (iter/update_step) == (iter//update_step) and iter >= 2*update_step:
            previous = np.mean(loss[iter-2*update_step:iter-update_step-1])
            actual = np.mean(loss[iter-update_step:iter-1])
            
            alpha = (actual/previous)*alpha
            
            loss_clustered.append(actual)
            
        elif (iter/update_step) == (iter//update_step) and iter == update_step:
            loss_clustered.append(np.mean(loss[iter-update_step:iter-1]))
            
        # Backward pass (Stochastic Gradient Descent)
        y_d  = [-1/y[i] for i in range(dim_out)]
        o_d  = [y[i] for i in range(dim_out)]
        o_d[c]-=1

        v_d = np.zeros((dim_hid,dim_out))
        h_d  = np.zeros(dim_hid)
        for j in range(dim_out):
            for i in range(dim_hid):
                v_d[i][j] += o_d[j] * h[i]
                h_d[i] += o_d[j] * v[i][j]       
        b2_d = o_d
        k_d  = [h_d[i]*h[i]*(1-h[i]) for i in range(dim_hid)]

        w_d  = np.zeros((dim_in,dim_hid))
        for j in range(dim_hid): 
            for i in range(dim_in): 
                w_d[i][j] = k_d[j] * x[i] 
        b1_d = k_d
        
        # Updting the weights
        for j in range(dim_hid):
            for k in range (dim_in):
                w[k,j] = w[k,j] - alpha*w_d[k,j]
            b1[j]= b1[j] - alpha*b1_d[j]

        for j in range(dim_out):
            for k in range(dim_hid):
                v[k,j] = v[k,j] - alpha*v_d[k,j]
            b2[j]= b2[j] - alpha*b2_d[j]
            
    print('mean loss of epoch',epoch+1,':',"%.4f" %np.mean(loss))
    print('\tmean loss of the last 100 iters:',"%.4f" %np.mean(loss[len(xtrain)-100:len(xtrain)-1]),'\n')
print('alpha started as 0.1, but now alpha =',"%.4f" % alpha)
fig, q4 = plt.subplots(figsize=(12, 6))
q4.plot(loss_clustered)
plt.title('Loss curve, update step: '+str(update_step))
plt.xlabel('iterations / '+str(update_step))
plt.ylabel('Loss')
plt.show()


# ### Implementing a neural network for the MNIST data (Question 5)
#     - I left the dynamic adaptative alpha and increased update_speed to 1000.
#     - I imported shuffle in order to randomize the positions of (xtrain,ytrain) for each epoch.

# In[26]:


# Importing libraries
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle # This one is used to shuffle both xtrain and ytrain
                                  #  without losing their conection
#Loading the MNIST data
(xtrain, ytrain), (xval, yval), num_cls = load_mnist()

# Normalization of the data
xtrain = xtrain / 255.0 # Since min = 0, max = 255
xval = xval / 255.0

# Dimensions of the neural network
dim_in = 784
dim_hid = 300
dim_out = 10

# Initialization
w  = np.random.randn(dim_in,dim_hid)
b1 = np.zeros(dim_hid)
v  = np.random.randn(dim_hid,dim_out)
b2 = np.zeros(dim_out)

alpha = 0.1
update_step = 1000 # Used in the loop that updates alpha and the clustered loss
loss_clustered = []

epochs = 5
for epoch in range(epochs):
    loss = []
    xtrainn,ytrainn= shuffle(xtrain,ytrain) # different order of the train values every epoch
    for iter in range(len(xtrain)):
        x=xtrainn[iter]
        c=ytrainn[iter]
        # Implementation of the NN
    
        # Forward pass
        k = w.T.dot(x) + b1
        h = 1/(1+np.exp(-k))
        o = v.T.dot(h) + b2
        
        soft = np.exp(o - o.max())
        y = soft / np.sum(soft, axis=0)
        
        # Loss
        l = -np.log(y[c])
        loss.append(l)
        
        # Alpha and clustered loss vector loop
        if (iter/update_step) == (iter//update_step) and iter >= 2*update_step:
            previous = np.mean(loss[iter-2*update_step:iter-update_step-1])
            actual = np.mean(loss[iter-update_step:iter-1])
            
            alpha = (actual/previous)*alpha
            
            loss_clustered.append(actual)
            
        elif (iter/update_step) == (iter//update_step) and iter == update_step:
            loss_clustered.append(np.mean(loss[iter-update_step:iter-1]))
            
        # Backward pass (Stochastic Gradient Descent)
        y_d  = -1/y
        o_d  = y
        o_d[c]-=1
        v_d = np.outer(h,o_d)
        b2_d = o_d
        h_d = np.matmul(v,o_d)
        k_d = h_d * h * (1-h)
        w_d = np.outer(x,k_d.T)
        b1_d = k_d
        
        # updating the weights
        w  -= alpha * w_d
        b1 -= alpha * b1_d
        v  -= alpha * v_d
        b2 -= alpha * b2_d

    print('mean loss of epoch',epoch+1,':',"%.4f" %np.mean(loss))
    print('\tmean loss of the last 1000 iters:',"%.4f" %np.mean(loss[len(xtrain)-1000:len(xtrain)-1]),'\n')
print('alpha started as 0.1, but now alpha =',"%.4f" % alpha)
fig, q4 = plt.subplots(figsize=(12, 6))
q4.plot(loss_clustered)
plt.title('Loss curve, update step: '+str(update_step))
plt.xlabel('iterations / '+str(update_step))
plt.ylabel('Loss')
plt.axis(ymax=2)
plt.show()


# ### Vectorized version of a batched forward and backward (Question 6)
#     I left static alpha and only kept track of the mean loss per epoch

# In[20]:


# Importing libraries
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle # This one is used to shuffle both xtrain and ytrain
                                  #  without losing their conection
#Loading the MNIST data
(xtrain, ytrain), (xval, yval), num_cls = load_mnist()

# Normalization of the data
xtrain = xtrain / 255.0 # Since min = 0, max = 255
xval = xval / 255.0

# Dimensions of the neural network
dim_in = 784
dim_hid = 300
dim_out = 10

# Initialization
w  = np.random.randn(dim_in,dim_hid)
b1 = np.zeros(dim_hid)
v  = np.random.randn(dim_hid,dim_out)
b2 = np.zeros(dim_out)

alpha = 0.1
loss_ep = []
epochs = 20

# Implementation of mini_bach (with batch_size = xtrain.shape[0] we get full batch)
batch_size = 50
x_mini_batches=[]
y_mini_batches=[]
n_batches=xtrain.shape[0]//batch_size
for i in range(n_batches):
    x_mini_batch=xtrain[i*batch_size:(i+1)*batch_size]
    x_mini_batches.append(x_mini_batch)
    
    y_mini_batch=ytrain[i*batch_size:(i+1)*batch_size]
    y_mini_batches.append(y_mini_batch)

if xtrain.shape[0]/batch_size != xtrain.shape[0]//batch_size:
    x_mini_batch=xtrain[(i+1)*batch_size:]
    x_mini_batches.append(x_mini_batch)
    
    y_mini_batch=ytrain[(i+1)*batch_size:]
    y_mini_batches.append(y_mini_batch)

for epoch in range(epochs):
    loss = []
    xtrainn,ytrainn= shuffle(x_mini_batches,y_mini_batches)
    for iter in range(len(xtrainn)):
        x=xtrainn[iter]
        c=ytrainn[iter]
        # Implementation of the NN
    
        # Forward pass
        k = np.matmul(x,w) + np.outer(np.ones(batch_size),b1)
        h = 1/(1+np.exp(-k))
        o = np.matmul(h,v) + np.outer(np.ones(batch_size),b2)
        
        y=[]
        for i in range(batch_size):
            soft = np.exp(o[i] - o[i].max())
            y.append( soft / np.sum(soft, axis=0) )
        y=np.array(y)
        
        # Loss
        l = [-np.log(y[i][c[i]]) for i in range(batch_size)]
        loss.append(np.mean(l))
        
        # Backward pass
        y_d  = -1/y
        o_d  = y
        for i in range(batch_size):
            o_d[i][c[i]]-=1
        v_d = np.matmul(h.T,o_d)
        b2_d = np.matmul(o_d.T,np.ones(batch_size))
        h_d = np.matmul(o_d,v.T)
        k_d = h_d * h * (1-h)
        w_d = np.matmul(x.T,k_d)
        b1_d = np.matmul(k_d.T,np.ones(batch_size))
        
        # updating the weights. We want the average of the gradients.
        w  -= alpha * w_d  / batch_size 
        b1 -= alpha * b1_d / batch_size 
        v  -= alpha * v_d  / batch_size 
        b2 -= alpha * b2_d / batch_size 
        
    loss_ep.append(np.mean(loss))    
    if epoch==0 or epoch == 19:
        print('mean loss of epoch',epoch+1,':',"%.4f" %np.mean(loss))
        print('\tmean loss of the last 50 iters:',"%.4f" %np.mean(loss[iter-50:iter-1]),'\n')
    elif (epoch+1)%5==0 :
        print('mean loss of epoch',epoch+1,':',"%.4f" %np.mean(loss))
axis=np.linspace(1,epochs,epochs)
plt.plot(axis,loss_ep)
plt.title('Loss curve, batch size: '+str(batch_size))
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()


# # Final experiments and analysis (Question 7)

# ### 1.- Training loss per epoch vs validation loss per epoch.

# In[21]:


# Importing libraries
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle # This one is used to shuffle both xtrain and ytrain
                                  #  without losing their conection
#Loading the MNIST data
(xtrain, ytrain), (xval, yval), num_cls = load_mnist(final=True)

# Normalization of the data
xtrain = xtrain / 255.0 # Since min = 0, max = 255
xval = xval / 255.0

# Dimensions of the neural network
dim_in = 784
dim_hid = 300
dim_out = 10

# Initialization
w  = np.random.randn(dim_in,dim_hid)
b1 = np.zeros(dim_hid)
v  = np.random.randn(dim_hid,dim_out)
b2 = np.zeros(dim_out)

alpha = 0.1

epoch_train_loss = []
epoch_val_loss = []
epochs = 5
for epoch in range(epochs):
    t_loss = []
    v_loss = []
    xtrainn,ytrainn= shuffle(xtrain,ytrain) # different order of the train values every epoch
    for iter in range(len(xtrain)+len(xval)):
        if iter < len(xtrain):
            x=xtrainn[iter]
            c=ytrainn[iter]
            # Implementation of the NN

            # Forward pass
            k = w.T.dot(x) + b1
            h = 1/(1+np.exp(-k))
            o = v.T.dot(h) + b2

            soft = np.exp(o - o.max())
            y = soft / np.sum(soft, axis=0)

            # Loss
            l = -np.log(y[c])
            t_loss.append(l)

            # Backward pass (Stochastic Gradient Descent)
            y_d  = -1/y
            o_d  = y
            o_d[c]-=1
            v_d = np.outer(h,o_d)
            b2_d = o_d
            h_d = np.matmul(v,o_d)
            k_d = h_d * h * (1-h)
            w_d = np.outer(x,k_d.T)
            b1_d = k_d

            # updating the weights
            w  -= alpha * w_d
            b1 -= alpha * b1_d
            v  -= alpha * v_d
            b2 -= alpha * b2_d
        else:
            x=xval[iter-len(xtrain)]
            c=yval[iter-len(xtrain)]
            k = w.T.dot(x) + b1
            h = 1/(1+np.exp(-k))
            o = v.T.dot(h) + b2

            soft = np.exp(o - o.max())
            y = soft / np.sum(soft, axis=0)

            # Loss
            l = -np.log(y[c])
            v_loss.append(l)
    print('Epoch '+str(epoch+1),'. Mean loss of training data:',"%.4f" % np.mean(t_loss),
          '| mean loss of validation data:',"%.4f" %np.mean(v_loss))
    epoch_train_loss.append(np.mean(t_loss))
    epoch_val_loss.append(np.mean(v_loss))
axis=np.linspace(1,5,5)
plt.plot(axis,epoch_train_loss,label='Training set', color='blue')
plt.plot(axis,epoch_val_loss,label='Validation set', color='red')
plt.title('Loss curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.axis(xmin=1,xmax=5,ymin=0)
plt.show()


# ### 2.- Average and standard deviation of multiple runs.

# In[136]:


# Importing libraries
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle # This one is used to shuffle both xtrain and ytrain
                                  #  without losing their conection
#Loading the MNIST data
(xtrain, ytrain), (xval, yval), num_cls = load_mnist()

# Normalization of the data
xtrain = xtrain / 255.0 # Since min = 0, max = 255
xval = xval / 255.0

# Dimensions of the neural network
dim_in = 784
dim_hid = 300
dim_out = 10

update_step = 100 # Used in the loop that updates alpha and the clustered loss

n_runs = 3
loss_clustered = []
for i in range(n_runs):
    loss_clustered.append([])
for run in range(n_runs):
    # Initialization
    w  = np.random.randn(dim_in,dim_hid)
    b1 = np.zeros(dim_hid)
    v  = np.random.randn(dim_hid,dim_out)
    b2 = np.zeros(dim_out)

    alpha = 0.1

    epochs = 5
    for epoch in range(epochs):
        loss = []
        xtrainn,ytrainn= shuffle(xtrain,ytrain) # different order of the train values every epoch
        for iter in range(len(xtrain)):
            x=xtrainn[iter]
            c=ytrainn[iter]
            # Implementation of the NN

            # Forward pass
            k = w.T.dot(x) + b1
            h = 1/(1+np.exp(-k))
            o = v.T.dot(h) + b2

            soft = np.exp(o - o.max())
            y = soft / np.sum(soft, axis=0)

            # Loss
            l = -np.log(y[c])
            loss.append(l)

            # Alpha and clustered loss vector loop
            if (iter/update_step) == (iter//update_step) and iter >= 2*update_step:
                loss_clustered[run].append(np.mean(loss[iter-update_step:iter-1]))

            # Backward pass (Stochastic Gradient Descent)
            y_d  = -1/y
            o_d  = y
            o_d[c]-=1
            v_d = np.outer(h,o_d)
            b2_d = o_d
            h_d = np.matmul(v,o_d)
            k_d = h_d * h * (1-h)
            w_d = np.outer(x,k_d.T)
            b1_d = k_d

            # updating the weights
            w  -= alpha * w_d
            b1 -= alpha * b1_d
            v  -= alpha * v_d
            b2 -= alpha * b2_d

        print('run '+str(run+1),'. Mean loss of epoch',epoch+1,':',"%.4f" %np.mean(loss))


# In[137]:


### getting the mean and standard deviation of the stacked loss vector.
loss_clustered = np.array(loss_clustered)
sigma = np.std(loss_clustered, axis=0)
mu = np.mean(loss_clustered, axis=0)

lower_bound = mu - sigma
upper_bound = mu + sigma

### Plotting the mean and the sigma region
t= np.linspace(1,len(loss_clustered[0]),len(loss_clustered[0]))

fig, ax = plt.subplots(1)
ax.plot(t, mu, lw=1, label='Mean loss', color='black')
ax.fill_between(t, lower_bound, upper_bound, facecolor='yellow', alpha=0.5,
                label='Standard Deviation')
ax.legend(loc='upper right')

ax.set_xlabel('iterations /'+str(update_step))
ax.set_ylabel('Loss')
ax.grid()


# ### 3.- SGD with different learning rates.

# In[30]:


# Importing libraries
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle # This one is used to shuffle both xtrain and ytrain
                                  #  without losing their conection
#Loading the MNIST data
(xtrain, ytrain), (xval, yval), num_cls = load_mnist(final=True)

# Normalization of the data
xtrain = xtrain / 255.0 # Since min = 0, max = 255
xval = xval / 255.0

# Dimensions of the neural network
dim_in = 784
dim_hid = 300
dim_out = 10

epoch_train_loss = []
epoch_val_loss = []
epochs = 5

for alpha in [0.03,0.01,0.003,0.001,0.0001]:
    # Initialization
    w  = np.random.randn(dim_in,dim_hid)
    b1 = np.zeros(dim_hid)
    v  = np.random.randn(dim_hid,dim_out)
    b2 = np.zeros(dim_out)

    epoch_train_loss = []
    epoch_val_loss = []

    for epoch in range(epochs):
        t_loss = []
        v_loss = []
        xtrainn,ytrainn= shuffle(xtrain,ytrain) # different order of the train values every epoch
        for iter in range(len(xtrain)+len(xval)):
            if iter < len(xtrain):
                x=xtrainn[iter]
                c=ytrainn[iter]
                # Implementation of the NN

                # Forward pass
                k = w.T.dot(x) + b1
                h = 1/(1+np.exp(-k))
                o = v.T.dot(h) + b2

                soft = np.exp(o - o.max())
                y = soft / np.sum(soft, axis=0)

                # Loss
                l = -np.log(y[c])
                t_loss.append(l)

                # Backward pass (Stochastic Gradient Descent)
                y_d  = -1/y
                o_d  = y
                o_d[c]-=1
                v_d = np.outer(h,o_d)
                b2_d = o_d
                h_d = np.matmul(v,o_d)
                k_d = h_d * h * (1-h)
                w_d = np.outer(x,k_d.T)
                b1_d = k_d

                # updating the weights
                w  -= alpha * w_d
                b1 -= alpha * b1_d
                v  -= alpha * v_d
                b2 -= alpha * b2_d
            elif epoch == epochs:
                x=xval[iter-len(xtrain)]
                c=yval[iter-len(xtrain)]
                k = w.T.dot(x) + b1
                h = 1/(1+np.exp(-k))
                o = v.T.dot(h) + b2

                soft = np.exp(o - o.max())
                y = soft / np.sum(soft, axis=0)

                # Loss
                l = -np.log(y[c])
                v_loss.append(l)
        if epoch == 0 :
            print('alpha = '+str(alpha))
        print('Epoch'+str(epoch+1),'. Mean loss of training data:',"%.4f" % np.mean(t_loss),
              '| mean loss of validation data:',"%.4f" %np.mean(v_loss))


# ### 4.- Final training and accuracy.

# In[45]:


# Importing libraries
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle # This one is used to shuffle both xtrain and ytrain
                                  #  without losing their conection
#Loading the MNIST data
(xtrain, ytrain), (xval, yval), num_cls = load_mnist(final=True)

# Normalization of the data
xtrain = xtrain / 255.0 # Since min = 0, max = 255
xval = xval / 255.0

# Dimensions of the neural network
dim_in = 784
dim_hid = 300
dim_out = 10

# Initialization
w  = np.random.randn(dim_in,dim_hid)
b1 = np.zeros(dim_hid)
v  = np.random.randn(dim_hid,dim_out)
b2 = np.zeros(dim_out)

alpha = 0.1

epoch_train_acc = []
epoch_val_acc = []
epochs = 5
for epoch in range(epochs):
    t_acc = 0
    v_acc = 0
    xtrainn,ytrainn= shuffle(xtrain,ytrain) # different order of the train values every epoch
    for iter in range(len(xtrain)+len(xval)):
        if iter < len(xtrain):
            x=xtrainn[iter]
            c=ytrainn[iter]
            # Implementation of the NN

            # Forward pass
            k = w.T.dot(x) + b1
            h = 1/(1+np.exp(-k))
            o = v.T.dot(h) + b2

            soft = np.exp(o - o.max())
            y = soft / np.sum(soft, axis=0)

            # Loss
            l = -np.log(y[c])
            t_loss.append(l)
            
            # Accuracy
            if abs(y[c]-max(y)) < 0.00001:
                t_acc+=1

            # Backward pass (Stochastic Gradient Descent)
            y_d  = -1/y
            o_d  = y
            o_d[c]-=1
            v_d = np.outer(h,o_d)
            b2_d = o_d
            h_d = np.matmul(v,o_d)
            k_d = h_d * h * (1-h)
            w_d = np.outer(x,k_d.T)
            b1_d = k_d

            # updating the weights
            w  -= alpha * w_d
            b1 -= alpha * b1_d
            v  -= alpha * v_d
            b2 -= alpha * b2_d
        else:
            x=xval[iter-len(xtrain)]
            c=yval[iter-len(xtrain)]
            k = w.T.dot(x) + b1
            h = 1/(1+np.exp(-k))
            o = v.T.dot(h) + b2

            soft = np.exp(o - o.max())
            y = soft / np.sum(soft, axis=0)

            # Accuracy
            if abs(y[c]-max(y)) < 0.00001:
                v_acc+=1
            
    print('Epoch '+str(epoch+1),'. Accuracy on training set:',str("%.2f" %(t_acc*100/len(xtrain)))+'%',
          '| Accuracy on validation set:',str("%.2f" %(v_acc*100/len(xval)))+'%')
    epoch_train_acc.append(t_acc*100/len(xtrain))
    epoch_val_acc.append(v_acc*100/len(xval))
axis=np.linspace(1,5,5)
plt.plot(axis,epoch_train_acc,label='Training set', color='blue')
plt.plot(axis,epoch_val_acc,label='Validation set', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.axis(xmin=1,xmax=5)
plt.show()


# In[48]:


axis=np.linspace(1,5,5)
plt.plot(axis,epoch_train_acc,label='Training set', color='blue')
plt.plot(axis,epoch_val_acc,label='Validation set', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.axis(xmin=1,xmax=5)
plt.show()


# In[49]:


acer = [3,4,5,4]
mu = np.mean(acer)
sigma = np.std(acer)
print(mu,sigma)

