**INSTALLATION GUIDE FOR SOFTWARE PROTOTYPE**

This guide describes how to install and run the software prototype across three different codes with step-by-step instructions: CNN\_B0, CNN\_B0\_SVM, and CNN\_B0\_SVM\_GENETIC. The first two codes are originally developed in Google Colab and the third code is developed in Visual Studio Code. The guide shows how to execute these codes in a local machine using Visual Studio Code coding plataform.

**REQUIREMENTS**

Ensure you have Python installed on your system. The recommended version is Python 3.8 or higher.

1. **Brain Tumor MRI Dataset:**
- Download dataset\_brain\_tumor\_mri dataset from [Github](https://github.com/DatAlbertW/Applied-Computational-Intelligence/tree/e001e722d32dac152964d0bff9ed1c23ea95777f/3.%20Software%20Prototype/dataset_brain_tumor_mri) or directly in [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
2. **General Packages:**
- pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
3. **Specific Packages:**
- **CNN\_B0\_SVM\_GENETIC:** pip install pygad plotly

**CODE MODIFICATIONS FOR LOCAL EXECUTION**

The code was developed in Google Colab, you need to modify it to run on a local machine. Specifically, remove or comment out this line related to Google Colab import and functionality:

- from google.colab import drive
- drive.mount('/content/drive')

**RUNNING THE CODE LOCALLY**

1. **EfficientNetB0 Prototype**
1) Download the code from [CNN_B0 Colab Link](https://colab.research.google.com/drive/1awAO03x2lRcNRjUxYktvX-TuRrQtvmg8?usp=sharing) or [GitHub](https://github.com/DatAlbertW/Applied-Computational-Intelligence/blob/9e3e17bd9e0f2a69ec62f15e8749a845f1f37941/3.%20Software%20Prototype/1_CNN_B0.ipynb).
1) Install the required packages as listed above. Modify the code to remove Google Colab-specific lines.
1) Correct path for ‘dataset\_brain\_tumor\_mri’
2. **Support Vector Machine Prototype**
1) Download the code from [CNN_B0_SVM Colab Link](https://colab.research.google.com/drive/1-WGyBD5c_iXHIIgJRjwv4u1UeiIbZQLU?usp=sharing) or [GitHub](https://github.com/DatAlbertW/Applied-Computational-Intelligence/blob/9e3e17bd9e0f2a69ec62f15e8749a845f1f37941/3.%20Software%20Prototype/2_CNN_B0_SVM.ipynb).
1) Install the required packages as listed above.. Modify the code to remove Google Colab-specific lines.
1) Correct path for ‘dataset\_brain\_tumor\_mri’ and the saved EfficientNetB0 model.
3. **Genetic Algorithm for Support Vector Machine Hyperparameter Optimization:**
1) Download the code from [CNN_B0_SVM_GENETIC GitHub Link](https://github.com/DatAlbertW/Applied-Computational-Intelligence/blob/9e3e17bd9e0f2a69ec62f15e8749a845f1f37941/3.%20Software%20Prototype/3_CNN_B0_SVM_GENETIC.ipynb).
1) Install the required packages listed above
1) Correct path for ‘dataset\_brain\_tumor\_mri’ and the saved EfficientNetB0 model.

**FINAL NOTES**

- The Pre-Trained EfficientNetB0 (‘efficientnetb0\_model.h5’) model Can be downloaded from [GitHub](https://github.com/DatAlbertW/Applied-Computational-Intelligence/blob/9e3e17bd9e0f2a69ec62f15e8749a845f1f37941/3.%20Software%20Prototype/efficientnetb0_model.h5).
- Ensure the paths to the **dataset\_brain\_tumor\_mri** image directory and to the pre-trained model are correctly set in the code.
