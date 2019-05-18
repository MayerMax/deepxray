# deepxray
Tools for medicine using the power of deep learning.

This project is dedicated to the medical problems we can potentially solve with the power of deep learning and computer vision.
This repository contains tools to simplify the process of developing systems using DL algorithms and useful facilities to maintain the stability of solutions: visual explanations of predictions, important metrics, and plots.

:exclamation: This is an __early stage__ of the project and not so many things are implemented.

When a new module or functionality will be added to the master branch of the project, the ``README`` will be updated to show the usage example.

### 1. Transfer learning
A classical approach to deal with a new task in computer vision is to use some architecture pretrained on a large dataset and replace original Fully-connected layers with a custom network and train only this part.

You can perform transfer learning using the functionality of ``DenseLearner`` class.

:warning: <br>
All the examples below are using private dataset created by the medical center [UGMK Health](https://www.ugmk-clinic.ru/page/translator/eng/). This dataset consists of ``14K+`` X-ray images of lungs in frontal and side orientations and annotated with a target class (``Normal`` if a patient has no tuberculosis and ``Pathology`` otherwise, the class ratio is highly unbalanced). Unfortunately, it is not possible now to make this dataset publically available.

```python
import pandas as pd
from keras.optimizers import RMSprop
from deepxray.transferlearning.dnn import DenseLearner

data = pd.read_csv('data.csv') # assuming this dataframe has target and image_path columns
train, val = train_test_split(data, stratify=data.target, test_size=0.2, shuffle=True, random_state=555)

size = (299, 299)
batch_size = 32
optimizer = RMSprop(lr=1e-4)

learner = DenseLearner(objective='binary', base='inceptionv3', layers=[512, 256], dropout=[0.5, 0.5])

history = learner.fit_from_frame(train, val, x_col='image_path', y_col='target', optimizer=optimizer, epochs=10)
learner.save('inception_base.pickle')

# unfreeze 5 last convolutional layers of the base network and fit the model again
history = learner.fit_from_frame(train, val, x_col='image_path', y_col='target', optimizer=optimizer, epochs=5, unfreeze=5)
learner.save('inception_base_unfreeze.pickle')

learner = DenseLearner.load('inception_base_unfreeze.pickle')
```

Now we are ready to make predictions. ``DenseLearner`` class has all the necessary methods to return predictions given path to image or dataframe with paths. To do this use ``predict_from_file`` or ``predict_from_frame`` methods.

In medicine, it is crucial to have a robust and explainable model. ``DenseLearner`` can plot a precision-recall curve to let you understand the capabilities of your solution and output a heatmap picture to explain why it predicted this or another class. It is also possible to manually choose threshold with binary problems to find the best properties with regard to your task.

```python
learner.plot_precision_recall_curve(val, x_col='image_path', y_col='target')
```
![precision-recall](https://i.imgur.com/EF57DWD.png)


To get the most probable class and the reason why the model predicts it, call the following method:
```python
path_to_test_image = 'test.bmp'
prediction, pil = learner.explain_prediction(path_to_test_image, filename='incept.bmp')
pil
```
![explanation](https://i.imgur.com/6FL5wJgm.png)

The model predicts ``Pathology`` class and correctly detect area with some blackout where tuberculosis is allocated.
``DenseLearner`` uses popular GRAD-cam algorithm for visual explanations and might applied to any conolutional layer.
 However, it might be not informative if the base network layers are not fine-tuned or trained from scratch due to the specificity of the data it was initially trained.
 
 ### 2. Demo application.
 In this repository, you can find a source-code for a demo web-based application for detecting tuberculosis from x-ray images of lungs.
 To launch it you need to clone a repository and locate all the files that ``DenseLearner`` object creates after calling to the ``save`` method.
 ```bash
 git clone https://github.com/MayerMax/deepxray.git
 cd deepxray\deepxray\diagnosis-support-application
 python server.py saved_model.pickle
 ```
 Visually it looks as following:<br><br>
 ![web-app](https://i.imgur.com/SIAXkJz.png)
 <br>
 
 :point_up: :point_up: :point_up: *Only for demonstration purpose!*
