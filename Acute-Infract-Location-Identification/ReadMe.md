# ACUTE INFARCT LOCATION PREDICTION

## PROBLEM STATEMENT:
Our aim is to classify the image into classes where each class indicates the region of infection of
Acute infarct disease in an optimised way using a Hardware Accelerator Openvino toolkit Provided by Intel.

## DATA:
Source of dataset: Prof Kousik Sankar

## DATASET
● 50 CASES
 - DWI.jpg
 - FLAIR.jpg
 - ADC.jpg
 
● PDF with mapping of Cases to Region of infection

## REQUIREMENT:
● Openvino
● Pandas
● Numpy
● Matplotlib
● Scikit-learn
● Keras
● Tkinter

## STEPS :

### RUN JUPYTER NOTEBOOK

- In a terminal or command window, navigate to the top-level project directory
ACUTE_INFRACT/ (that contains this README) and run one of the following
commands:
- ipython notebook Acute_infarct.ipynb
Or
- jupyter notebook Acute_infarct.ipynb
- This will open the Jupyter Notebook software and project file in your browser.

### RUN THE INFERENCE IMAGE IN COMMAND PROMPT
Convert the .pb file created from jupyter notebook into .xml and .bin file using the following
steps:
- [1] cd C:\Program Files(x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer
- [2] python mo_tf.py --input_model "path of .pb file" --output_dir "destination path for .xml" --model_name mri_infer --input_shape [1,32,32,3]
Setup openvino variables
- [3] cd C:\Program Files (x86)\IntelSWTools\openvino\bin
- [4] setupvars.bat
For running inference file
- [6] python mri_predict.py -i "path to inference image" -d "CPU" -m "path to .xml file"

### CONCLUSION:
A model with train accuracy of 92% and test accuracy of 42% is achieved with small dataset.
