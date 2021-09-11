# Facial Recognition System

Designed to recognize employees of a company and thus give authorization. The system also consists of a module that detects the faces of an image and returns a crop of the image in order to facilitate recognition. The system is based on sending images of workers to a neural network and training it, then every time someone wants to access, the camera captures an image, goes through the face detection module and then through the face recognition module , which is responsible for deciding whether the individual is authorized.

## Dataset

We decided to use the Olivetti faces dataset. This dataset consists of 400 images of faces of 40 people, which are 10 images of each person. The images are only cutouts of the faces, are 64x64 in size and are grayscale. In order to test the system, what we will do is assume that these 40 people are the workers of the company, so it will be with these images that we will train the model initially.

## Model

The main part of this project is based on the model that decides whether or not a person is an employee of the company. Therefore, the system used by neural networks, based on different layers of neurons
to make the final decision is what interests us.
It was decided to use a sequential CNN model because you did not want to send more than one input at a time and you did not want the layers to share neurons, so each layer is separate and has an independent input and output. Three convolutional layers are used through which the data passes through filters. In the first two layers the images go through 32 filters and in the third they go through 64. Each layer is followed by a layer of Max Pool2D that adapts the output data to the input data in the next layer and finally has put a layer of Dropout at the end, which puts some units to 0 at random, this process serves to prevent overfitting. Finally, the model is run using the Adam optimization algorithm.

## Detect Faces 

We used a previously trained face detection model to be able to obtain the position of the faces in images. Once the position of the face in the image has been obtained, the image is cropped and later the relevant checks are made. These image clippings are adapted to match the data in the dataset used.

## Execution Flow

<img width="873" alt="diagrama_flux" src="https://user-images.githubusercontent.com/65408666/120235441-49723a00-c25a-11eb-9434-4c06c5e871e6.png">

## Tests and Results

### Parameters of the Neural Network
To achieve a good result we have been varying the following parameters:

-Learning Rate: The learning rate controls the rate at which the descent gradient adapts to the data, so it must be modified to reach a learning rate that gives a good result.
-Epoch: This is the number of iterations that the model has to train. If the learning rate becomes smaller then the number of iterations must be increased because the model is learning more slowly.
-Dropout: Dropout is the number of units that are discarded and set to 0 at each time of training.
-Threshold: When deciding whether an image contains an authorized person, a threshold has been set that decides whether or not the person in the image is authorized based on their match percentage. Depending on the results of the training, the threshold used varies.
-Training dataset size: In the tests, the percentage of data in the training dataset and in the test dataset was also varied.

The combinations that have given better results have been:

<img width="329" alt="Captura de pantalla 2021-05-31 a las 21 51 46" src="https://user-images.githubusercontent.com/65408666/120235482-67d83580-c25a-11eb-8f89-a31d15b8d156.png">

Graphic with 120 epochs:
<img width="278" alt="Captura de pantalla 2021-05-31 a las 21 58 29" src="https://user-images.githubusercontent.com/65408666/120235901-56dbf400-c25b-11eb-87f4-f80dbe390b52.png">

Graphic with 60 epochs and 80% train:
<img width="278" alt="Captura de pantalla 2021-05-31 a las 21 59 10" src="https://user-images.githubusercontent.com/65408666/120235945-6eb37800-c25b-11eb-9726-c4cced9a745c.png">



## Execution guide

pip install -r llibreries.txt

For launch the app, run 'main.py'
