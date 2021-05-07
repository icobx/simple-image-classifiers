# gender & attractiveness classifiers
 
- simple **MLP** and **SVM classifiers** for determining gender and attractiveness from detected face.
- works on *live* images from *camera*
- accuracies of models trained on images embedded using [*resnet*](https://github.com/timesler/facenet-pytorch "Facenet-pytorch repository") model
    - MLP accuracy | gender: ***98%***, attractiveness: ***74%***
    - SVM accuracy | gender: ***93%***, attractiveness: ***71%***
- there are also models trained on images embedded using **LBP** (local binary patterns) technique, but with disappointing results
- images which are being embedded are normalised (aligned and resized) and are subset of [**CelebA**](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset
- done as one of bi-weekly school assignments

#### some images:

- Confusion matrices for MLP classifier<br>
![Confusion matrix, MLP, attractiveness](/docs/cm-mlp-deepEmb-attr.png "Confusion matrix, MLP, attractiveness")<br>
![Confusion matrix, MLP, gender](/docs/cm-mlp-deepEmb-male.png "Confusion matrix, MLP, gender")

- Confusion matrices for MLP classifier<br>
![Confusion matrix, SVM, attractiveness](/docs/cm-svm-deepEmb-attr.png "Confusion matrix, SVM, attractiveness")<br>
![Confusion matrix, SVM, gender](/docs/cm-svm-deepEmb-male.png "Confusion matrix, SVM, gender")

- Demonstration of classification working on images from camera<br>
![Classification on images from camera](/docs/cam.png "Classification on images from camera")

