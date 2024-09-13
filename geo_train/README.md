# Beggiatoa Coverage Estimation
## INTRODUCTION
![](screenshots/DenseNet121_11cls_cls6_stat.png) </br>
⬆ Figure 1. coverage estimation result of DenseNet121 </br>
<p align='justify'>Beggiatoa is a bacterium that is also an anoxic and polluted environment with percentage coverage in the benthic area beneath pens used in some regions for compliance monitoring [1]. In order to do this, it is first necessary to score images or video beneath pens based on the appearance and cover of different types of Beggiatoa [2]. To automate the process, image classification with patch-base inference &amp; training algorithm is used. </p>

## DATA OVERVIEW
The target classes are **6 classes** like following picture:
![](screenshots/data_annotation_original.png) </br>
⬆ Figure 2. original classes of given data. </br>
<p align='justify'>However, the dataset categorized into 6 classes are not capable of representing full spectrum of under water biodiversity. Because the circumstances of under water environment are varied and it is hard to define the types of under water situataion into strictly 6 types. So, in this project, I have subcategorized some of the 6 classes into more detailed 12 classes according to their texture feature like following: </p>

![](screenshots/data_subcategory.png)</br>
⬆ Figure 3. Subcategorized classes.</br>
<p align='justify'>When you see Figure 3 - type 3 and background classes, the instances are homogeneous in their appearance and image feature. Including those homogeneous images in the same class impedes DNN model training results. Therefore, it is reasonable to seperate those into subcategorized class so that it can be trained as if different class. But don't worry, they will be merged into original class in the inference stage.</p> </br>

![](screenshots/data_division.png) </br>
⬆ Table 1. Data division overview </br>
To guarantee the reliability of training and validation result, the data is divided in mutually exclusive way. </br>

## TRAINING SCHEME
![](screenshots/training_scheme.png) </br>
⬆ Figure 6. Training scheme overview </br>

# RESULT
## Visualized result
![](screenshots/inference_example.gif) </br>

<p align='justify'>The first column of the table below is the inference result of the trained model with 11 classes. The second column of the table is the final result with merged subcategorized class into corresponding classes which is the actual target result. </p>

Model| detailed classification      |  final result
:------:|:---------------------------------------------------:|:-------------------------:
Xception    |![](screenshots/Xception_11cls_11cls.png)  |  ![](screenshots/Xception_11cls_cls6.png)
Inception ResNetV2   |![](screenshots/InceptionResNetV2_11cls_11cls.png)  |  ![](screenshots/InceptionResNetV2_11cls_cls6.png)
EfficientNetB0   |![](screenshots/EfficientNetB0_11cls_11cls.png)  |  ![](screenshots/EfficientNetB0_11cls_cls6.png)
EfficientNetV2B0  |![](screenshots/EfficientNetV2B0_11cls_11cls.png)  |  ![](screenshots/EfficientNetV2B0_11cls_cls6.png)
DenseNet121    |![](screenshots/DenseNet121_11cls_11cls.png)  |  ![](screenshots/DenseNet121_11cls_cls6.png)

⬆ Figure 5. Test results of coverage estimation upon the trained five models.

## Evaluation result class basis
![](screenshots/per_class_eval.png) </br>
⬆ Table 2. Evaluation result of trained models on validation dataset </br>
<p align='justify'>As you can see from the Table 2, morphologically obvious classes such as 6_number and 5_big_worm have shown high accuracy in all model. On the other hand, 2_pathcy, 3_thick and 3_thin which are ambiguous and hard to be distinguished classes have shown relatively low accuracy score.</p>

## Training Process
![](screenshots/evaluation.png) </br>
⬆ Table 3. Evaluation result of the best trained model on training and evaluation dataset </br>
</br>
![](screenshots/Training_loss.png) </br>
⬆ Figure 6. Training loss changes during training process of the five models</br>
 </br>
![](screenshots/Training_accuracy.png) </br>
⬆ Figure 7. Training accuracy changes during training process of the five models</br>
 </br>
![](screenshots/Validation_loss.png) </br>
⬆ Figure 8. Validation loss changes during training process of the five models</br>
 </br>
![](screenshots/Validation_accuracy.png) </br>
⬆ Figure 9. Validation accuracy changes during training process of the five models</br>
 </br>

## Discussion
* How to deal with overlapped patches?
* How to deal with edge object?

## TO DOs
* implementation of interface for quantitative analysis
* 5 fold cross validation (done 19th/Feb) -> data cleansing needed. (done 23/Feb)
* Apply Voting algorithm for more fine-edge classification (done)

## Code base 
https://github.com/boguss1225/classification-patch-base

## REFERENCES
[1] Crawford, C., Mitchell, I., Macleod, C.: Video assessment of environmental impacts of salmon farms. ICES J. Mar. Sci. 58(2), 445–452 (2001) </br>
[2] Yanyu Chen, Yunjue Zhou, Son Tran, Mira Park, Scott Hadley, Myriam Lacharite, and Quan Bai. 2022. A Self-learning Approach for Beggiatoa Coverage Estimation in Aquaculture. In AI 2021: Advances in Artificial Intelligence: 34th Australasian Joint Conference, AI 2021, Sydney, NSW, Australia, February 2–4, 2022, Proceedings. Springer-Verlag, Berlin, Heidelberg, 405–416. https://doi.org/10.1007/978-3-030-97546-3_33
