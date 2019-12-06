# Text to Image Generation using DC-GANs

# Problem Statement
<b>Proposal: <br></b>
In the world of computer vision, natural language descriptions can be tricky in automatically generating images. It is still a fundamental problem to generate fine-grained high-quality images just based on text descriptions. With the help of Generative Adversarial Networks (GANs), this project reviews and represents  methodology that have been implemented to <b>generate realistic images based on text descriptions</b>

# DataSet

*   The model is currently trained on the flowers dataset. Download the images from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/ and save them in Data/flowers/jpg. 
*   The dataset consists of 8189 images of 102 flower categories
*   Each image has ten text captions that describe the image of the flower in different ways. For that, download the captions from https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view. Extract the archive, copy the text_c10 folder and paste it in Data/flowers
*   The code to unzip the dataset is available under UnzipData.ipynb

# DataSet Split Strategy
*   We split the dataset into distinct training and test sets. It has 82 train+validation and 20 test classes. We used 5 captions per image for training. 
*   The training image size was set to 32 * 32 * 3. We have compressed the images from 500 * 500 * 3 to the set size so that the training process would be fast.

# Architecture & Implementation
<b>We have used deep convolutional generative adversarial network (DC-GAN) flavor of GAN networks in this project</b> Below are the list of main code files used in the project along with their brief description:
### dcgan.py:
This class has functions to create a model, load the model and fit model.
### img_cap_loader.py:
This class loads images and categories for text and creates an array of the corresponding image and text.
### glove_loader.py
GloVe, coined from Global Vectors, is a model for distributed word representation. The model is an unsupervised learning algorithm for obtaining vector representations for words. This is achieved by mapping words into a meaningful space where the distance between words is related to semantic similarity. Training is performed on aggregated global word-word co-occurance statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. It is developed as an open-source project at Stanford.
### dcgan_train.py:
The main class which loads the data, trains the model using the data, saves the model and also generates snapshots during training.
### dcgan_generate.py:
This class has function to load images as well as text and normalize them for the GAN model.

# Run Environment & System Configuration

<b>We run our entire deep learning model under Google Cloud platform (GCP)</b><br> Below are the steps we took to achieve the same. Several screenshots are also provided below supporting the work done.
*   Setup Google account using free $300 credits provided by Google
*   Create a new project
*   Deployed <b>deep learning virtual machine having 1 GPU of type NVIDIA Tesla T4 and 16vCPUs with 104 GB RAM</b>
*   Initialized Google SDK on our machines to create an SSH connection
*   Using deployment manager we hit the localhost:8080 button to lauch the Jupyter Lab GUI within GCP and ran our entire code there

# Generated Images and Sample Results

<h4><b>Images generated after running 150 epochs:</h4></b>
<img src="https://imgur.com/gctmz9h.png" alt="Screen Demo" width="750" />

<h4><b>Actual Images generated after running 1000 epochs:</h4></b>
<img src="https://imgur.com/uYFxSnA.png" alt="Screen Demo" width="750" />
