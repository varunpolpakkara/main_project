# Logo Classification

This code performs image classification on the Flickr 27 Logos dataset for first 10 Logos using SIFT Feature Extractor and Kmeans Clustering for extracting significant features which are then fed into a Linear SVM Detector.

This code is developed on the concept of bag of Visual Words Model, and these features extracted are clustered into words using Kmeans Clustering.

This model's accuracy can be increased by using convolutions to detect features with at least 5 layers but that couldn't be trained due to lack of computational power.

Also The training accuracy is __100%__ using Linear SVM at C = 1.0

The number of clusters created are 400 i.e. the number of significant features.

Data augmentation can further increase accuracy, but has not been implemented because Kmeans Clustering hangs due to increased dataset.

The testing performed is on the query set on full images.

Run `python extract.py` to extract features <br/>
Then `python train.py` to train the Linear SVM Detector <br/>
Finally `python test.py` to test the trained Detector <br/>
