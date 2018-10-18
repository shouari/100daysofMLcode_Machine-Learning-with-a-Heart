# 100daysofMLcode_DrivenData Challenge: Heart Diseases Presence

Data set provided by https://www.drivendata.org.

Sources are listed in scripts.

If any comment or improvement suggetion please feel free, all suggestions are welcom


This my very first chanllenge in ML, and I really enjoy.

I started Andrew Ng courses 5 weeks ago on www.coursera.com and following Siraj Raval (https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A) and Sentdex(https://www.youtube.com/user/sentdex/)  Channels.


# Introduction:

The goal of the challenge is to is to predict whether or not a patient has heart disease given historical data (https://www.drivendata.org/competitions/54/machine-learning-with-a-heart/page/107/).

Apart from the data (csv files) and submission format (csv), there are 04 python scripts:

**1- normalize_data.py**: the script aims to normalize the data set, it is a class that can be imported later in the training script and submission script.

**2- heart_disease_nn.py** : The neural network script it contains
        
      _predict :  Method to predict the output of a given dataset
      _accuracy : Evaluate accuracy
      _train_model : to train model, after training model is saved in "modelname.pickle" and can be                                                   called later
      _test_model : test the trained model.
                            
 The network designed is a 4 layers, with adjustable nodes number for hidded layer. The output layer is 01 node since we need to predict the probability for a patient to have a heart disease (0 being healthy, 1 sick)
 
 For the activation functions, during the subission I used Relu for the hidden layers and sigmoid for output layer.
 However Tanh function is defined in the script so it is possible to test the behiavor of the network with other activativation.
 
 As for the loss function, as per the challenge description, log loss function is used 
 
     **Log loss =−1n∑ni=1[yilog(y^i)+(1−yi)log(1−y^i)]**
                            
I will explain all the steps I went through for the design of the layer and "mistakes" I did before reaching and acceptable result and list down all references.

  **3- training.py** : the scipt calls the ***NeuralNetwork** from ***heart_disease_nn.py** and ***_normalize*** methode from ***normalize_data.py***.
 
 Parameters of the network of the network are adjusted and the model is saved in "modelnam.pickle"
 
  **4- submission.py** :  The script simply load the saved model, upload the submission data set and push the results to submission.csv file as per the challenge formatting.
                            
To test the script, all is need is to open ***training.py*** input the paramet of the nework and then go ***submission.py*** to save the result in the csv file.

