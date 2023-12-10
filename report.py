import streamlit as st
st.set_page_config(page_title="Multi-Class-Segmentation-report", initial_sidebar_state='collapsed')

st.markdown("## Multi-Human Parsing via Pixel-Accurate Segmentation of Body Parts and Fashion Items")
st.markdown('')

st.markdown("Shreyas Bangera")
st.markdown('')

st.markdown("### **Abstract**")

st.markdown("""
    This report presents a comprehensive approach to addressing the Multiclass Segmentation Task, focusing on the detailed
    segmentation of human body parts and fashion items in a diverse dataset of human parsing images. The dataset consists of
    images with multiple individuals, each represented by separate annotation files. The segmentation task involves 
    identifying and outlining various elements, including head, torso, arms, legs, and various fashion items such as hats,
    clothes, and shoes.

    The segmentation process follows standard protocols for human body parts and fashion items, emphasizing accurate outlining
    and labeling. An essential aspect of my approach involves handling multiple labels for each image, where hair, sunglasses
    and other accessories are shared labels among individuals. I address this challenge by applying element-wise addition to 
    consolidate unique segments from individual masks into a unified mask.

    The evaluation of my approach employs pixel accuracy and Intersection over Union (IoU) as metrics, yielding 89% accuracy on both metrics. 
    The report discusses the dataset's intricacies, challenges faced during segmentation, and the strategies employed to overcome these challenges.

    The deliverables include segmented images from the dataset, a comprehensive report detailing the segmentation 
    process, challenges, and solutions, along with folders containing evaluation images with segmentation masks.
    The report provides valuable insights into the segmentation of complex, multi-individual images.
            
""")


st.markdown('')
st.markdown("### Table of Contents")
st.markdown("""
    - Introduction
    - Data Familiarization
    - Handling Multiple Annotation Files
    - Loading and Preparing the Data
    - Model Selection
    - Preliminary Experiment
    - Fine-Tuning Iteration 1
    - Fine-Tuning Iteration 2
    - Scope for Future Work
    - References
""")

st.markdown('')
st.markdown("### Introduction")
st.markdown("""
    The primary goal of this task is to perform detailed segmentation of the given dataset. This dataset consists of 
    multiple human parsing images and masks, focusing on segmenting individual human body parts and fashion items from 
    various images. Task Description: Data Familiarization: Begin by reviewing the given dataset to understand the variety
    of images, masks and the level of detail required for segmentation. Segmentation Guidelines: Follow the standard 
    segmentation protocols for human body parts and fashion items. This includes, but is not limited to, separating 
    individual elements like head, torso, arms, legs, as well as clothing items such as shirts, pants, dresses, etc. 
    Quality Assurance: Ensure that each segmented part is accurately outlined and labeled. Pay special attention to 
    the edges and overlapping items. Evaluation Criteria: Accuracy of segmentation and labeling. Expected Deliverables: 
    Segmented images from the given dataset, with each body part and clothing item accurately outlined and labeled. A 
    comprehensive report detailing the segmentation process, challenges faced, and how they were overcome. A zip file 
    containing all evaluation images with segmentation mask.
    
    Details about the dataset and folder structure.
            
    :blue[./images]: All images in the dataset.
             
    :blue[./annotations]: The segmentation annotation files corresponding to the images. 
    One image is corresponding to multiple annotation files with the same prefix, one file per person. 
""")
st.markdown("""
    In each annotation file, the label represents:
""")
data = {
    'Class Values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 , 14, 15, 16, 17, 18],
    'Class Names': [
        'background', 'hat', 'hair', 'sunglass', 'upper-clothes', 'skirt', 'pants', 'dress', 'belt',
        'left-shoe', 'right-shoe', 'face', 'left-leg', 'right-leg', 'left-arm','right-arm', 'bag', 'scarf', 'torso-skin'           
    ]
}

st.dataframe(data)
st.markdown("""
    :blue[./visualization.m]: Matlab script to visualize the annotations
            
    :blue[./train_list.txt] The list of images for training 
    and validation
            
    :blue[./test_list.txt] The list of images for testing.
""")

                    

st.markdown('')
st.markdown("### Data Familiarization")
st.markdown("""
    
    The images folder contains people ranging from single to many. The annotations however are separately stored in different 
    files for each person in an image. For example, consider an image with filename 0001.jpg, having 3 people. 
    The annotations/mask files for this image are 0001_03_01.png, 0001_03_02.png and 0001_03_03.png where the first 
    4 numbers reference the image, the next two indicate the number of people in the image and the last one reference a 
    specific person in the given image. The nature of data and labels is such that hair of all the people in the 
    image is represented with the number 2 and sunglasses of all the people with 3, rather than assigning three different 
    values to three different people. What changes is the body parts or accessories within a person not across persons.  
""")
st.markdown('')
st.markdown("### Handling Multiple Annotation Files")
st.markdown("""
    Here is how I approached multiple labels (one file per person) for a given image. The background is represented with number
    0 and other segments like hat, hair, sunglasses, clothes are represented as 1,2,3,4 etc. A key observation I made is that
    only 0 has overlapping pixels in all of the related annotation files. No other numbers have overlapping pixels in a given
    set of related annotation files. So, I applied element wise addition to stitch all unique segments from all the individual
    annotations into a consolidated annotation file.
""")

st.markdown('')
st.markdown("### Loading and Preparing the Data")
st.markdown("""
    The initial configuration (No. of images) of dataset was as follows:
""")
data = {'Training Set': [4000], 'Test Set': [980]}
st.dataframe(data)
st.markdown("""
    The dataset was then restructured to the following configuration:
""")
data = {'Training Set': [3500], 'Validations Set': [500], 'Test Set': [980]}
st.dataframe(data)
st.markdown("""
    A `CustomDataset` class subclassed from `Dataset` class in Pytorch was created to load images and annotations into 
    the training environment as per the train-validation-test split. These dataset objects were then translated into `DataLoader` 
    objects for batch training.
""")

st.markdown('')
st.markdown("### Model Selection")
st.markdown("""
    DeepLabV3 with a ResNet50 backbone originally pretrained on the COCO dataset was chosen as the starting point for training.
    DeepLabV3 is a state-of-the-art deep learning architecture designed for semantic segmentation, and using ResNet-50 as a 
    backbone provides a good balance between model complexity and computational efficiency. Some key features that makes DeepLabV3
    well suited for segmentation tasks are Atrous (Dilated) Convolution, ASPP (Atrous Spatial Pyramid Pooling), Backbone Network (ResNet-50)
    Pretrained Weights (COCO_WITH_VOC_LABELS_V1).
            
    DeepLabV3 pretrained model has three main components in its network architecture.
    1. backbone
    2. classifier
    3. aux_classifier
            
    All the parameters of the 'backbone' network were frozen and the rest of the model parameters were fine-tuned on the
    provided dataset.
            
""")

st.markdown('')
st.markdown("### Preliminary Experiment")
st.markdown("""
    A preliminary run was performed with a small fraction of the data to get insights into the model performance. 
    The dataset for this run consisted on 16 training images and 4 validation images. For this particular experiment, the
    input size was [4, 19, 520, 520] and target/annotation size was [4, 520, 520]. The model seemed to learn from the data
    as evident from the convergence shown in Fig. 1. 
""")
st.image('experiments/0/performance.png', caption='Fig. 1. Loss Convergence and Pixel Accuracy.')


st.markdown('')
st.markdown("### Fine-Tuning Iteration 1")

st.markdown("""
    This section describes various stages of the fine-tuning performed after the preliminary run.
    Full dataset with the configuration outlined in the 'Loading and Preparing the Data' section was used in the iteration. 
            
    **Fine-Tuning environment**
    - Kaggle's Jupyter Notebook
    - Device: GPU Tesla P100-PCIE-16GB
            
    **Issues encountered while loading full data.**
    - A corrupt/truncated file was found in the dataset.
    - This caused the training loop to break down while loading data from `train_loader`.
    - A function was created to detect the truncated file and necessary changes to the PIL ImageFile settings were made.
    - One or more images were non-three channelled and caused issues while normalizing the data.
    - This issue was solved by converting all images to RGB in the `__getitem__` method of the `CustomDataset` class.
            
    **Image Preprocessing steps**
    - Convert to RGB using PIL
    - Resize to (512, 512) using Bilinear interpolation
    - Convert to PyTorch Tensors
    - Normalize with mean and standard deviation values of [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] respectively.
            
    **Annotations Preprocessing steps**
    - Resize to (512, 512) using Nearest interpolation
    
    **Hyperparameters used during training**
    - Train batch size: 32
    - Validation batch size: 32
    - Test batch size: 32
    - Input image size: (3, 512, 512)
    - Learning Rate: 0.01
    - Optimizer: Adam
    - Loss function: main Cross-Entropy Loss + 0.4 * aux Cross-Entropy Loss
    - Training epochs: 10 (110 training batches)
    - Number of output units: 2
    - Layer freezing: Backbone network frozen.
""")
st.markdown("""
    **Before vs. After**
            
    A simple and intuitive side-by-side comparison of image and predicted masks are shown below. Fig. 2 shows random images from the
    dataset and its corresponding predicted masks before fine-tuning. Fig. 3 shows the same after fine-tuning.
""")
st.image('experiments/1/pretrained-mask-output.png', caption='Fig. 2. Random images and outputs before fine-tuning')
st.divider()
st.image('experiments/1/fine-tuned-mask-output.png', caption='Fig. 3. Same random images and outputs after fine-tuning')

st.markdown('')
st.markdown('**Performance**')
st.image('experiments/1/performance.png', caption='Fig. 4. Performance of the model in iteration 1')

data = {
    'Metric': ['Pixel Accuracy', 'IoU'],
    'Training': ['91.29%', 'Not measured'],
    'Validation': ['88.25%', '89.02%'],
    'Test': ['88.00%', '89.04%']
}
st.dataframe(data)

st.markdown("**Conclusions**")
st.markdown("""
    - Consistent scores across splits show model generalizes well instead of just memorizing train data.
    - No significant overfitting evident as validation and test metrics are quite close to training.
    - Alignment between pixel accuracy and IOU metrics indicates spatially coherent segmentation.
    - Achieving ~89% in both key measures on final test set implies strong real world performance.
    - On inspecting some of the images and predicted masks, the output revealed that classes like sunglasses, scarfs and similar
      parts that are relatively rare or cover relatively smaller area of the images were partially or completely missing in
      the predicted masks. This could be the side-effect of class imbalance in the dataset; not everybody wears scarfs, hats
      and sunglasses but most have hair and other features.
    - We also see some of the features like clothes, limbs etc., being partially present in predicted masks. This could
      arise from the inherent complexity of data. The complexity could arise from noise, textures in the
      background and occlusion.
""")
st.markdown('')
st.markdown('**Model Checkpoint 1**')
st.markdown("""
    The model weights obtained after iteration 1 of fine-tuning is called :orange['deeplabv3_resnet50_finetuned_v1'] with a '.pt' extension.
""")


st.markdown('')
st.markdown('### Fine-Tuning Iteration 2')
st.markdown("""
    This iteration is about further fine-tuning the saved model checkpoint (deeplabv3_resnet50_finetuned_v1) with exactly the
    same configuration as earlier except now it is fine-tuned for 5 epochs.
""")
st.markdown('**Performance**')
st.image('experiments/2/performance.png', caption='Fig. 5. Performance of the model in iteration 2')
st.markdown("""
    The train and validation losses as evident from Fig. 5. are diverging (validation loss is increasing) from each other hinting
    that the model is overfitting to the training data. Accuracy of the other hand seems to be increasing on the training set however, on the validation set,
    it is quite stagnant.
""")
st.markdown('')
st.markdown('**Same random images and predicted masks**')
st.image("experiments/2/finetuned-mask-output.png", caption="Fig. 6. Same random images and outputs after iteration 2 of fine-tuning")
st.markdown('**Conclusions**')
st.markdown("""
    - Since overfitting was observed in iteration 2 of fine-tuning, this checkpoint :orange['deeplabv3_resnet50_finetuned_v2'] was not chosen
    for final segmentation task.
""")

st.markdown('')
st.markdown('### Scope for Future Work')
st.markdown("""
    Earlier the problem of incomplete segmentation, missing pieces were discussed and how factors like noise, background texture 
    and occlusion could possibly limit the performance of deep learning models on segmentation tasks. We also stated a few possible
    factors that are intrinsic to the dataset such imbalanced classes and accuracy of annotation iteself. Based on the discussion
    so far, here are a some suggested areas for future work.

    1. Handling class imbalance with weighted loss function such that less frequent classes are given more importance.
    2. Extensive Hyperparameter Search if resources are available.
    3. Expanding the dataset with diverse training examples.
""")

st.markdown('')
st.markdown('**Class Imbalance in the Data**')
st.image('experiments/3/class-imbalance.png', caption="Fig. 7. Class imbalance in the dataset")
st.markdown('')
st.markdown('**Pixel-Wise Class Frequency in the Dataset**')
data = {
    'Class Name': [
        'background', 'hat', 'hair', 'sunglass', 'upper-clothes', 'skirt', 'pants',
        'dress', 'belt', 'left-shoe', 'right-shoe', 'face', 'left-leg', 'right-leg', 
        'left-arm','right-arm','bag','scarf', 'torso-skin'
    ],

    'Frequency': [
        582349311, 2129907, 21327800, 382582, 121018505, 7292041, 62997029, 35287623, 964656,
        6082579, 6062071, 21558699, 10596836, 10584134, 10556918, 10518277, 3596278, 744594, 3454160
    ]
}
st.dataframe(data)

st.markdown('')
st.markdown('**Suggested Class Weights for Weighted Loss Function.**')

data = {
    'Class Name': [
        'background', 'hat', 'hair', 'sunglass', 'upper-clothes', 'skirt', 'pants',
        'dress', 'belt', 'left-shoe', 'right-shoe', 'face', 'left-leg', 'right-leg', 
        'left-arm','right-arm','bag','scarf', 'torso-skin'
    ],

    'Class Weight': [
        0.08292219686429116, 22.67220315747416, 2.2641662154805613, 126.22048138837246, 0.3990272744695228, 6.622245296005099, 0.7665390729859073,
        1.3684595363798326, 50.05896838927692, 7.939014719007565, 7.965872423883903, 2.2399164351488148, 4.556990804663422, 4.562459641055784,
        4.574221776708535, 4.591026097765472, 13.427683902781242, 64.85371116410596, 13.980152688504967]
}
st.dataframe(data)

st.markdown('')
st.markdown('### References')
st.markdown("""
    1. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
    2. [Torchvision Documentation](https://pytorch.org/vision/stable/index.html)
""")