import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.neural_network
numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1))
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
loss=np.zeros(20)
f1=np.zeros((20,2))
for neurons in range(1,21):
    mlp = sklearn.neural_network.MLPClassifier(activation='logistic', 
                                               hidden_layer_sizes=(neurons),max_iter=1000)
    mlp.fit(x_train, y_train)
    loss[neurons-1]=mlp.loss_
    f1[neurons-1,0]=sklearn.metrics.f1_score(y_test, mlp.predict(x_test), average='macro')
    f1[neurons-1,1]=sklearn.metrics.f1_score(y_train, mlp.predict(x_train), average='macro')
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(range(1,21),loss)
plt.xlabel("# de neuronas")
plt.ylabel("Loss")
plt.subplot(1,2,2)
plt.plot(range(1,21),f1[:,0])
plt.plot(range(1,21),f1[:,1])
plt.ylabel("F1")
plt.xlabel("# de neuronas")
plt.legend(["F1_test","F1_train"])
plt.savefig("loss_f1.png")
plt.show()
scale = np.max(mlp.coefs_[0])
plt.figure(figsize=(12,30))
for i in range(10):
    plt.subplot(5,2,i+1)
    plt.imshow(mlp.coefs_[0][:,i].reshape(8,8),cmap=plt.cm.RdBu, 
                       vmin=-scale, vmax=scale)
    plt.title("neurona %3.0f"%i)
plt.savefig("neuronas.png")
plt.show()