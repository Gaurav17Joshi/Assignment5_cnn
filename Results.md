|                    |    VGG 1   |    VGG 3   |    VGG 3 DA   |    VGG 16 TL   |
|:-------------------|:----------:|:----------:|:-------------:|---------------:|
|Training Time       |  4.636 s   |  25.80 s   |  26.09 s      |  86.02 s       |
|Training Loss       |  0.623     |  0.392     |  0.512        |  0.088         |
|Training Accuracy   |  89.38     |  83.75     |  80.62        |  98.75         |
|Testing Accuracy    |  67.50     |  75.00     |  77.50        |  95.00         |
|Number of Parameters|  12.8,6    |  12.9,6    |  12.9,6       |  138,6         |

Number of trainable parameters in transfer learned vgg16 = 8194 (4096*2)

Discussion on Results:-
Q1) Are the results as expected? Why or why not?
A1) Yes, the results are as expected, the vgg model performs well on categorising images. Our data set was a difficult one as apples and guavas have similar shapes, and while apples are red and guavas are green, there are also green apples and ripe guavas, when cut open are red in color. Keeping this fact in mind, the cnns performed progressively well with more sophisticted models giving even better tests accuracies.

Q2) Does data augmentation help? Why or why not?
A2) Data agumentation helps in creating a better model for prediction, as now the images send into the model are not center aligned and rotated at angles or flipped, keeping the image similar but changing it orientation. The models training accuracy decreases, as now it have to even more varied images, but it now is better able to identify even disoriented images, which may well occur in many of the test dataset images.

Q3) Does it matter how many epochs you fine tune the model? Why or why not?
A3) Yes, the number of epochs on which the model is trained is very important as, if we train on a high number of epochs , the network gets the same images again and again, and learns to identify them well (training accuracy reaches 100), but the test accuracy drops as the model, is highly trained on identifying the patterns of the test set, which removes its focus form the overall important features of the classes, (say if train apples have leaves on their tip, then it gives a high weight to a leaf, whihc may not be the case for training images). This is very similar to bias variance tradeoff, say in decision trees, where increasing the depth leads to better training accuracy but worser test accuracy.

Q4) Are there any particular images that the model is confused about? Why or why not?
A4) Yes, the images where the guava is cut open and the only the open part is shown is tough to classify as apple or guava, also, the image of clipart apples and 

eg:-
apple/apple fruit28
apple/apples19
apple/apple fruit77
