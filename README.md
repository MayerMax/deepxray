# deepxray
Tools for medicine using the power of deep learning.

This project is dedicated to the medical problems we can potentially solve with the power of deep learning and computer vision.
This repository contains tools to simplify the process of developing systems using DL algorithms and useful facilities to maintain the stability of solutions: visual explanations of predictions, important metrics, and plots.

:exclamation: This is an __early stage__ of the project and not so many things are implemented.

When a new module or functionality will be added to the master branch of the project, the ``README`` will be updated to show the usage example.

### Simple transfer learning
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
