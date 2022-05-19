# We created this GitHub project to facilitate editors and reviewers to review our paper "Automatic lung segmentation in chest X-ray images using improved U-Net"

Instructions：


The codes are stored in a folder called "ab"

1.runing "data_generator.py"  (Load the required lung field masks dataset)

2.Using "date augment.py" can improve the generalization ability of the model

3."model.py" is the main program of neural network, which is written with tensorflow 2.4.0

4.The lung field output from the model is a floating-point two-dimensional matrix, so you need to use "best"_ Threshold. Py "get the best threshold. Binarize the predicted value and compare it with the real lung field.

5."five metric.py" (these are five commonly used lung segmentation evaluation indexes)

6."Computer graphics morphology algorithm. Py" includes connected domain filtering and flooding


If you want to get the detailed code, please contact lwf@haut.edu.cn
