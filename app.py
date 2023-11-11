import streamlit as st

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

from skorch import NeuralNetClassifier
import torch

from models.nn import CreateModel

@st.cache_data
def load_mnistData():
    mnist = fetch_openml('mnist_784',as_frame=False,cache=True)
    return mnist

mnist = load_mnistData()

print(mnist.data.shape)

totalData = mnist.data.shape[0]

st.write("hi but this is confusing as shit")

# Data preprocessing:

X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# Normalising

X = X/ 255.

st.write("The total size of the dataset is: "+str(totalData))

selectedData = st.slider("DATA SAMPLES ",1000,10000,value=5000,step=1000)

st.write('''This data needs to be divided into training and testing data. 
        The model will use the training data during the training step, and the testing data will not be seen by the model''')
st.write("lets split into data into train and test sets, by default as seen in the below slider the split is set as 70% for training\
         and 30% for testing. It is a known knowledge that more training data will imporve the model performance as long as the data is \
         more different")
sliderValue = st.slider("choose training data percentage",10,100,step=5,value=70)


def processData(X,y,selectedData=2000,ratio=0.7):
    Xre = X.reshape(-1, 1, 28, 28)
    X_train,X_test,y_train,y_test = train_test_split(Xre[:selectedData],y[:selectedData],test_size=ratio,random_state=42)
    assert(X_train.shape[0]+X_test.shape[0] == selectedData)
    return X_train,X_test,y_train,y_test
    

selectedAct = st.radio("activations",options=["ReLU","Sigmoid","Tanh"],index=0)
selectedDepth = st.slider("depth of layers",1,10,step=1,value=10)

# processData(X,y,selectedData,sliderValue/100)

def loadModel(in_channels,selectdAct,selectedDepth):
    newModel = CreateModel(in_channels,selectdAct,selectedDepth)
    return newModel

epochs = st.slider("epochs",1,100,step=1,value=10)

pressed = st.button("Train model",type="primary")

@st.cache_resource
def trainModel(X_train,y_train,epochs=10):
    torch.manual_seed(0)
    st.write('the data lenght is: ',selectedData)
    st.write("The epoch is ",epochs)
    model = loadModel(1,selectedAct,selectedDepth)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = NeuralNetClassifier(model,max_epochs=epochs,lr=0.1,device=device)
    with st.spinner("Training in progress...",cache=False):
        net.fit(X_train,y_train)
    return net

X_train,X_test,y_train,y_test = processData(X,y,selectedData,sliderValue/100)

print("pressed state = ",pressed)
if pressed:
    trainedModel = trainModel(X_train,y_train,epochs)

predict = st.button("Test Model",type="primary")

if predict:
    trainedModel = trainModel(X_train,y_train,epochs)
    preds = trainedModel.predict(X_test)
    score = accuracy_score(y_test,preds)
    st.success("the model predicted with an accuracy of :"+str(score))
    st.write('Accuracy with the current config: '+str(score))







