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
A1) Yes, the results are as expected, 

Q2) Does data augmentation help? Why or why not?
A2)

Q3) Does it matter how many epochs you fine tune the model? Why or why not?
A3)

Q4) Are there any particular images that the model is confused about? Why or why not?
A4)
