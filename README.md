**1. Executive Summary**
* Project Title: Caribbean and Hawaiian Marine (Reef) Life Image Classification
* Name: Ryan T. Johnson
* Date: May 4, 2025

Brief Overview:
When snorkeling in the Caribbean and Hawaiian reef ecosystems, I have often found breathtaking views of many unique marine life. Still, I am at a loss when identifying the species I encounter underwater. It is difficult to determine what exactly I am looking at in the reef life ecosystems. I have some ideas, but I am at a loss as a non-expert. Many of the different marine life creatures look similar, which can make it harder for me to identify them correctly. Traditionally, guides and books are used to figure out what you have seen in the water, which can be time-consuming and impractical when discovering a wide variety of marine life creatures in the ecosystem. This makes it especially hard for people snorkeling to figure out what they see underwater. Based on this challenge, there is a growing need for an easier, faster, and more convenient way to identify different marine life species from underwater images. 

To overcome and solve this challenge, I built an image classification model, using deep learning techniques to automatically classify three marine life species in the Caribbean and Hawaiian reef ecosystems they are:
![The three marine life species I am using](https://github.com/user-attachments/assets/060952ae-8502-433b-88d7-12431b2acc2c)

I personally captured over 1000 underwater images with my GoPro camera, then labeled and trained the dataset on a deep learning image classification model to accurately classify each distinct species, using tools like TensorFlow, Keras, and Roboflow to create the dataset. To improve the model's overall performance, I implemented techniques such as image augmentation, normalization, and other image enhancement techniques to ensure that the model can handle various conditions of the images captured underwater. I had to account for multiple real-world conditions to ensure the model could perform adequately. 

Once the results for the model were trained, it achieved strong performance in correctly identifying the different species. The **accuracy is approximately 83%** based on the unseen test images. The solution helps users gain a deeper appreciation for the reef life ecosystem through image classification. It shows how artificial intelligence can be a valuable tool for identifying distinct marine life. My recommendations moving forward include expanding the data set to include more marine species to include more variety in the model, employing a web-based application to allow different users to deploy the model in a friendly and easy-to-use platform that will enable snorkelers, divers, and others to upload underwater images and receive real-time species identification for the aquatic life they discover. Other recommendations include gathering more marine life images to help grow the dataset and improve overall species diversity so that the model can be used in a wide variety of underwater adventures and finally collaborate with marine biologists and/or conservation programs in providing adequate identification for the new range of marine life images collected through observing the marine (reef) life underwater.


**2. Introduction**
* Background:
  * As an avid snorkeler with a deep appreciation for the unique marine ecosystems of the Caribbean and Hawaiian reefs, I have spent time exploring reefs and captured enormous amounts of videos and images underwater using my GoPro. Still, I am always at a loss for what I am taking an image of; some marine life creatures I know quite well, including a Green Sea Turtle, but others I am at a complete loss in identifying. The image classification model will create a system in which I can upload distinct sea creatures, and the model can properly identify and allow me to know what exactly I am looking at without needing a guide key and books to identify different underwater animals that exist. The image classification model is a way to speed up the comprehension of underwater creatures and identify sea creatures when snorkeling in ocean ecosystems with my GoPro, which takes enormous amounts of pictures and videos.

* Problem Statement:
  * Reef ecosystems in the Caribbean and Hawaiian regions are home to diverse marine species that are vital in maintaining oceanic ecological balance. Despite underwater pictures being captured during snorkeling excursions, identifying marine reef species remains a significant challenge for non-experts. Traditional identification methods include guidebooks and manual comparison. These are time-consuming, require prior knowledge, and often fail to provide immediate answers on the distinct marine life the snorkelers are looking at. Many snorkelers and marine enthusiasts cannot fully understand or appreciate the biodiversity (marine life) they encounter during their explorations. There is a need for an automated, accurate, and user-friendly system to classify marine species from underwater images and enhance real-time understanding of reef ecosystems. That is where a deep learning image classification model comes into play; it can automatically identify the different marine life species from underwater images and allow the user to understand the marine life ecosystems below the water. 	

* Objectives:
  * **Develop an image classification model** using Convolutional Neural Networks (CNN) to identify marine reef life species from underwater images captured during snorkeling. 

  * **Build a custom annotated dataset for image classification** from GoPro footage collected during my snorkeling excursions in the Caribbean and Hawaiian regions, ensuring proper class labeling and image preprocessing.

  * **Classify three distinct marine life species** in the Caribbean and Hawaiian reef ecosystems: Green Sea Turtle, Eagle Ray, and Stingray. Note: I had to remove Cuttlefish and Reef Triggerfish because the image datasets were limited.

  * **Automate the identification process** of marine reef life to eliminate the need for manual guidebooks, species keys, and manual identification while exploring underwater environments.

**3. Data Description**
* Data Source:
  * A custom-labeled dataset was developed using Roboflow
  * Manually annotated each marine life image into one of three classes: **Eagle ray**, **Green Sea Turtle**, or **Stingray**.
  * Roboflow was used for manual labeling.
  * Roboflow's API was used to download the dataset to GitHub
  * No external datasets (e.g., Kaggle) were used.

* Data Collection:
  * Photos and videos were captured while snorkeling in the **Caribbean and Hawaiian reefs between 2017 and 2025** using a **GoPro** underwater camera.
  * Roboflow was used to strip the videos into frames per second, which allowed images to be extracted from a video.
  * Initially, five different marine life species classes were Cuttlefish, Eagle ray, Green Sea Turtle, Reef triggerfish, and Stingray. Still, Cuttlefish and Reef triggerfish were removed due to the limited number of images used for the testing dataset. The two classes were causing issues with the performance of the model.

* Data Preprocessing Techniques:
  * Data Normalization 
    * Scales the input data from 0 to 255 into a range between 0 and 1 by dividing each pixel value by 255
    * Normalization ensures that the numbers are similar in size so that everything fits into a range between 0 and 1
    * Normalization helps with the training process and overall performance of the model.
  * Data Augmentation
    * Image enhancement techniques applied are flipping, rotational tilt, zoom quality, and contrast, which change the light and shadow.
    * Improves the model's performance by creating additional images for the training dataset
    * Strengthens the model for different real-world conditions in underwater photos

* Data Overview:
  * The dataset size is **1,030 images**
  * Marine Life Class Distribution:  
    * Eagle ray: 366 images
    * Green Sea Turtle: 482 images
    * Stingray: 182 images.
  * The file format for the dataset is JPG.
  * Dataset split:
    * Training: 728 images
    * Validation: 201 images
    * Testing: 101 images
  * Image Dimensions: 500 × 500 pixels
  * RGB colors are used, which have three channels: red, green, and blue
  * Batch Size: 32, which means that the model processes 32 images at a time during each training iteration
  * Set a seed to reproduce results
  * Annotation Type: Single-label classification (one class per image).
  * Labeling Method: Manually labeled 1,030 images.
  * Roboflow API is used to download the dataset into GitHub.


**4. Methodology**
* Approach:
  * A **supervised learning** approach was used for the image classification model.
  * The image classification model used is a **Convolutional Neural Network (CNN)** with **single-label classification**.
  * Each image is classified into one of these three classes: **Eagle ray**, **Green Sea Turtle**, or **Stingray**.
  * The CNN model learns distinguishing features of the images during training and predicts the correct marine life species for unseen test images.
  * The CNN model classifies the distinct marine life species based on the entire image.

* Model Selection:
  * **Convolutional Neural Network (CNN)** is the ideal approach when creating an image classification model because it can automatically learn and extract the features of an image.
  * The CNN has a customizable architecture that allows the developer to create a model for complex marine life images.
  * Artificial Neural Networks (ANNs) were considered, but I ultimately found that the CNN model worked best for the image classification model because it can automatically extract features from photos and is best for complex classification of images.
  * My familiarity with CNNs and their effectiveness for image classification models made them an optimal choice for the Caribbean and Hawaiian marine reef life classification project.

**5. Results** 
* Key Findings:
    * The **accuracy is 83%** based on 101 unseen test images.
    * Data augmentation reduced the model loss and improved from the original 0.60 to 0.51.
    * Data augmentation didn't improve; the new accuracy is 0.81 compared to the original 0.86.
    * The model classified most images correctly, with minor confusion between Green Sea Turtles and Eagle rays and between Stingray and Green Sea Turtles.
    * The **Eagle Ray** marine life class has the strongest performance and the highest F1 score out of the other classes.
* Performance Metrics:
**Training Results Overview:**
Evaluation of Training Results of the CNN Model **Before** Data Augmentation  
![Before](https://github.com/user-attachments/assets/f25048fe-9fec-42ae-ae6e-a9f71e437a05)

**Note:** An epoch is one cycle of training for the model.

**The Left Plot is the Model Loss:**
In the model loss plot on the left, we see that both the training and validation losses are decreasing over epochs, which indicates training is occurring. The loss fluctuates for the validation loss after epochs 6 and 7; the loss increases at epochs 10 and 12. At epoch 10 for the validation loss, there is a spike due to a possible overfitting moment that occurred while training the model.

**The Right Plot is the Model Accuracy:**
The training and validation accuracies improve over each run and training epoch in the model accuracy on the right. It peaks from epochs 8 to 10 for the validation accuracy and then slightly dips. The model accuracy for training is around 0.85, and the validation is around 0.86. Both of these accuracies are close together. It is interesting to note that validation accuracy is higher than training accuracy.

* The validation loss is around 0.60, and the accuracy is around 0.86.

Evaluation of Training Results of the CNN Model **After** Data Augmentation
![after](https://github.com/user-attachments/assets/b234b59d-793f-4967-80bd-a81a4a8769e9)

**Note:** An epoch is one cycle of training for the model.

**The Left Plot is the New Model Loss with Data Augmentation:**
The model loss for training and validation loss steadily decreases after each epoch run. The validation loss had small fluctuations, but overall, the loss decreased. No overfitting occurred for the training and validation loss curves.

**Right Plot is the New Model Accuracy with Data Augmentation:**
In the model for accuracy for training and validation, we see that they both increase. The validation accuracy peaks between epochs 9 and 11 and finishes above the training accuracy. The validation accuracy fluctuates slightly around epoch 10 but trends upwards. The training accuracy ends around 0.80, and the validation accuracy ends around 0.80. Overall, the model performed excellently.

* The validation loss with data augmentation is around 0.51, and the accuracy with data augmentation is around 0.81.

**The Predictions of Caribbean and Hawaiian Marine Life Based on Unseen Test Dataset Images**
![predictions](https://github.com/user-attachments/assets/2c86c143-658d-4c83-b3cf-1685fbf0403a)

Based on the images above, we can see that most of the marine life species were classified correctly, but a few stingrays and Green Sea Turtles were misclassified. Overall, it is incredible to see how well the Convolutional Neural Network (CNN) model performed; by seeing these visualizations, the user can clearly understand how well the model performed in classifying each marine life species in a dataset.

**Note:** 
* **True Labels** are the original species class of each image.
* **Predicted Labels** are the classification predictions the CNN model makes for each image.
* The **Confidence Score** shows how confident the model is with the prediction made for each image.


#### Classification Metrics Overview:

* **Loss** measures how poorly the model's predictions are compared to the actual labels. **A lower loss indicates better model performance.**
* **Accuracy** measures the percentage of correct predictions out of total predictions.


**Test Loss and Accuracy Analysis:**
* Test Loss: 0.413 
* Test Accuracy: 0.832
* The accuracy is around 83% based on the unseen test dataset, which strongly determines how well the model performed.

**Confusion Matrix Analysis:**
The confusion matrix shows the model's performance and how well it works. 
![matrix](https://github.com/user-attachments/assets/08f2344c-6696-43d3-8970-54f84d49a831)

Confusion Matrix Explained
* **Eagle ray:** 35 images were correctly identified. There is one misclassified as a Green Sea Turtle. Other than that, Eagle Ray performed nearly perfectly.
* **Green Sea Turtle:** 39 images were correctly classified; 8 were misclassified as Eagle Ray, and one was misclassified as Stingray. Other than some confusion with Eagle Ray and Stingray, it was a solid overall performance. For the Green Sea Turtle, the most confusion was being misclassified as the Eagle Ray.
* **Stingray:** 10 images were correctly classified, and seven were misclassified as Green Sea Turtle. All were confused with Green Sea Turtles.

**Classification Report Analysis:**
The classification report is a summary of how the model performed. The report tells the user how well the model classifies marine life species.

**The Classification Report Metrics:**
* **Precision:** Measures the percentage of the correct positive predictions compared to the total positive predictions made by the model.
* **Recall:** Measures the percentage of the correct positive predictions compared to the total actual positive predictions that the model finds.
* **F1-score:** Combines the precision and recall into a single metric, which shows how well the model performed. The higher the F1-score is, the closer it is to 1, which indicates how well the model performed. F1-score is based on precision and recall.
* **Support:** Shows the number of true instances for each class in the test dataset. It shows the total number of images for each of the classes.
![report](https://github.com/user-attachments/assets/8e89bc07-bf75-410f-b5b6-5fc4f86cfa88)

Classification Report Explained
* **Eagle Ray** has a recall of 0.97, and the precision is slightly lower at 0.81, likely from false positives from Green Sea Turtles. The F1-score is 0.89; the Eagle Ray had a strong performance overall.
* The **Green Sea Turtle** has a high precision of 0.83 and a recall of 0.81 because the model misclassified some Green Sea Turtles with other classes. The F1-score is 0.82, which is good but has room for improvement.
* **Stringray** has a precision of 0.91, and the recall is 0.59 because the model misclassified some instances. The F1-score is 0.71, the lowest of the other classes, which needs improvement.

**Results Continued:**
* Interpretation: 
  * **Data augmentation** did help reduce overfitting by improving the model's loss, but it didn't improve the test accuracy. This suggests that data augmentation made the model more generalized, but is limited to the performance gains.
  * The CNN model successfully identified **distinct marine life species with reasonable accuracy**.
  * **Surprising insight of confusion matrix:** The Eagle Ray was correctly identified the most in the model compared to the Green Sea Turtle. Since the Green Sea Turtle class had more images in the dataset, I thought it would be classified the best out of all the other marine life classes, but it wasn't.
  * Even though the model's performance was not perfect, it identified the distinct marine life species for snorkelers and explorers of the reef ecosystems. 
  * The results show that AI and computer vision can be practical tools in supporting the exploration of marine life species; future improvements are needed to perfect the image classification model.


**6. Discussion**	
* Limitations: 
  * **Limited images for the distinct classes** are one of the most significant limitations I encountered, as I did not have enough images for the image classes for the dataset. Initially, the dataset included five marine life classes: Cuttlefish, Eagle ray, Green Sea Turtle, Reef Triggerfish, and Stingray. However, I had to **remove Cuttlefish and Reef Triggerfish** because each class **only had 6 to 7 images in the test dataset**, which caused poor performance and misclassification to occur more frequently. To improve the accuracy and reliability of the performance, I had to remove these two classes from the model. I may add them back when I capture more Cuttlefish and Reef Triggerfish images, but that is in my future work.

  * The **quality of underwater images** is another limitation of the images collected for the dataset. The underwater images' lighting, water clarity, and shadowing conditions made seeing some distinct marine life species difficult. Since the project aims to simulate realistic experiences for snorkelers and marine life enthusiasts, I didn't use photo enhancement tools like Photoshop or other photo editing software; I wanted to keep the original images as natural as possible for the model to distinguish. As a result, it may have made it more difficult for the model to process and understand how to identify distinct marine species predictions correctly.

  * A **single-label classification approach** is used for the model, which means that only one species can be identified per image. However, while snorkeling, multiple species can appear together in images in other scenarios, and a **multi-label classification or object detection** model should be more appropriate for those cases. These approaches are often time-consuming and require more resources. Since I had to create the dataset for the model manually, single-label classification was the most practical solution for the project's scenario.


* Challenges:
  * **Integrating the Roboflow dataset with my GitHub repository** was a significant challenge early on. I could not save and commit the dataset I would use for the image classification model. Eventually, I resolved the issue using the **Roboflow API**, which allowed me to use code to download the dataset into GitHub's project environment, an effective solution.

  * **Training efficiency and resource management** for the deep learning image classification model was the challenge I ran into, ensuring that the model trained effectively and prevented overfitting. I used **EarlyStopping**, which halted the training after four epochs (training cycles) without improvement. The maximum epochs (training cycles) that the model could train is up to 25, but oftentimes, the model effectively ran on fewer. The EarlyStopping technique helps reduce unnecessary training cycles and ensures that computing resources are used effectively.

  * **Hardware constraints** were another challenge I encountered while training the image classification model. I used a **CPU** to train the model due to hardware limitations. A GPU is preferred for training the image classification model because of the complex computer processing often required for image classification models. The CPU completed the CNN model's training in under 20 minutes despite this.

  * After the initial training of the model, **multiple misclassification issues** were encountered when classifying the original five marine life classes included. The misclassification negatively impacted the precision and recall scores and downgraded the overall accuracy. Ultimately, the model improved overall performance by **removing Cuttlefish and Reef Triggerfish**, which were the lowest performing and often caused the most confusion. These adjustments made the model more effective in learning the three remaining marine life species: **Eagle Ray, Green Sea Turtle, and Stingray**.


* Ethical Considerations:
  * The image classification project contains no data, including human subjects or personal data, so the **privacy concerns are minimal**. I collected all the images using a GoPro camera while snorkeling in the Caribbean and Hawaii on excursions. **No personal information on humans is included.**

  * Adding new marine life classes to the dataset in the future can cause marine life species misclassification, which could potentially misinform the users about what marine life species they are identifying. The user should be advised that I am not a marine biologist expert by any means and that the model is **not a substitute for expert identification**. The **model should not be the primary source for scientific or ecological identification**. The user should consult with a marine life expert. The model is intended to be an assisting tool for snorkelers.

  * Environmental responsibility is another ethical consideration that promotes responsibility and how to interact with the reef ecosystems. In the future deployment of the model, information about **ethical snorkeling practices should be included to avoid irreversible harm to the reef habitats**. No human should damage the reef ecosystem, which marine life species rely on to live; protecting the reef ecosystem is essential.


**7. Recommendations & Next Steps**
* Actionable Insights:
  * The **accuracy is 83%** and shows reliable performance in classifying the marine life species: **Eagle rays, Green Sea Turtles, and Stingrays**.
  * The image classification model has a strong potential for snorkelers, drivers, and marine life enthusiasts to use the model in **real-world scenarios to identify reef species**.
  * A user-friendly web-based or mobile application should be deployed to help users upload their underwater images and receive real-time predictions on the distinct marine life species they are finding.

* Future Work:
  * **Expand the dataset to include more marine species**, providing more variety to the model and enhancing the range of marine life used beyond three classes.
  * **Develop a web-based or mobile application** to allow different users to deploy the model in a friendly and easy-to-use platform that will enable snorkelers, divers, and others to upload underwater images and receive real-time species identification for the aquatic life they discover.
  * **Gathering more marine life images** will help grow the dataset and improve overall species diversity so that the model can be used in various underwater adventures. Consider using crowdsourcing to help develop the dataset.
  * **Collaborate with marine biologists and/or conservation programs** to provide adequate identification for the new range of marine life images collected by observing underwater marine (reef) life.
  * **Incorporate multi-label classification and object detection (YOLOv8)** into the model to discover multiple marine life species within a single image. Currently, only one species is identified per photo; this would allow an image with various species to be identified effectively. 


**8. Conclusion**
* Summary of Impact:
  * The project successfully demonstrated how Caribbean and Hawaiian Marine life species from underwater images can be automatically identified using **deep learning and convolutional neural networks**. The model achieved a strong accuracy of 83% on unseen data, which relays how the model is a powerful tool that snorkelers, divers, and marine life enthusiasts can effectively identify and understand the biodiversity of the marine life that lives underwater. The project combines the aspects of underwater exploration and the ability to identify reef life through a faster and more accurate approach compared to traditional identification methods. 

* Final Thoughts:
  * As an avid snorkeler, I am photographing marine life without knowing what I am capturing. I always need help identifying the specific aquatic wildlife I discover. Image classification is an excellent tool in which the model trained can identify distinct marine life quickly to help others identify them. Identifying and understanding marine life species can lead to exploring the ecological balance in our oceans. Many different fascinating creatures can be discovered through research and observation. The more information an individual has, the better they will want to explore other unique aquatic wildlife.


**9. References**
* Sources:
  * DSCI 619 Deep Learning Course Materials
  * DSCI 598 Capstone Project Course Materials
  * https://roboflow.com/
  * https://roboflow.com/annotate
  * https://app.roboflow.com/protozoa/caribbean-and-hawaii-marine-life/2
  * https://www.tensorflow.org/tutorials/images/classification
  * https://www.geeksforgeeks.org/convolutional-neural-network-cnn-in-machine-learning/
  * https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns
  * https://www.splunk.com/en_us/blog/learn/data-normalization.html
  * https://docs.github.com/en/get-started/start-your-journey/hello-world
  * https://github.com/desktop/desktop/issues/18225
  * https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
  * https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
  * https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten
  * https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
  * https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
  * https://www.statology.org/sklearn-classification-report/
  * https://asana.com/resources/executive-summary-examples
  * https://www.tutorialspoint.com/cnn-vs-ann-for-image-classification 


**10. Appendices**
A. Technical Modeling Details

**Convolutional Neural Network (CNN) Model Architecture**
In the convolutional layers, we use Conv2D() for performing the CNN with kernels(filters), padding='same' is used to keep the output size the same as the input, and activation='relu' is used to learn about patterns by using ReLU (Rectified Linear Unit) non-linearity. Then, we also used MaxPooling2D() to downsample a window size 2x2. These are the first steps that allow the CNN model to be created. The number of filters used is from 16 to 128 in the convolutional layers. **Note:** I didn't use filters up to 256 because that can lead to overfitting.

The other layers in the CNN model are
* Flatten Layer- used to convert the multi-dimensional input data from the output of the convolutional layers to a one-dimensional vector, which is then used by the dense layer
* The dense layer has 256 neurons that are used to connect the flattening layer to the dense layer, which is then used to extract the combinations used in the beginning (Conv2D() and MaxPooling2D()). I also used activation='relu' again to learn about the relationships.
* Dropout layer - I added this to help prevent overfitting; it randomly drops 50% of neurons to improve training.
* Output layer- has three classes (3 neurons based on the labels), which are the three different marine life species. I also included activation='softmax' in the output layer to ensure that each output is between [0, 1] and to help with probabilities. The output layer gives us the predictions for the five classes for the input image.

By creating each of these layers, the neural network can learn from the training data to predict what each image is and the class(label) they align with, making predictions on the marine life images. Overall, CNN learns how to classify the Caribbean and Hawaiian marine life images based on patterns found during training.

**Additional Technical Details**
For more technical details and code, please visit the **GitHub Repository Caribbean and Hawaiian Marine(reef) Life Single-Label Image Classification**. The repository includes a Jupyter notebook that outlines all the code with comments and details on how the model was developed and tested.
