# Code for 'Anisotropy of Metamaterials' book

**Author**: Radoslaw A. Kycia

**Description**: This code contains complete examples to the chapters:
- 'Digital Image Processing and basic granulometry' - directory: *1_DigitalImageProcessingAndBasicGranulometry*
- 'Cancer cells detection using Neural Networks' - directory: *2_CancerCellsDetectionUsingNeuralNetworks*

of the book 'Anisotropy of Metamaterials'.

The subdirectories are enumerated and named to associate them to specific sections in the book.


**Usage**: Each subdirectory contains a python file/files containing complete code from a given section. In each example there is *output* directory that will contain the output of the Python script after running the code. It was cleaned to save the disc space, since you can generate output on your own.

In directory *2_CancerCellsDetectionUsingNeuralNetworks* you need to create model first using the code from *1_NNModelPreparation*. First you have to download and unzip to train and test datasets from [PCam dataset](https://github.com/basveeling/pcam), namely:
- camelyonpatch_level_2_split_test_x.h5.gz
- camelyonpatch_level_2_split_test_y.h5.gz
- camelyonpatch_level_2_split_valid_x.h5.gz
- camelyonpatch_level_2_split_valid_y.h5.gz
They are large, so they were removed to save the space in the repository.
Then you copy the model from *1_NNModelPreparation/model/* to *2_ImageProcessing/model/* directory and run the python file from *2_ImageProcessing* directory. We left the model since it is small, however you can try to train it on your own.


**Prerequisites**: You need Python 3.x interpreter and a lot of standard libraries for Data Analysis, Image Processing and Deep Learning:
- numpy
- pandas
- matplotlib
- sklearn
- opencv
- tensorflow
- ...
Just try to run the script and if there will be Python *import error* just install required packages.

**Warning**: The code is by no means novel. On the contrary, it is a standard basic code that can be invented in one weekend, when you learn Image Processing and related topics in granulometry. So, possible, it will be 'boring' for experts. The code was written in elementary style (without advanced Software Engineering concepts) to be easy to grasp by novices in the field and Scientists which are no experts in advanced programming. The code was written having good intentions in mind and was testes, however its proper works depends on many factors including usage and configuration, and therefore, we are not responsible for any damage you can do by running it. Use at your own risk.


**Acknowledgements**: If you find this examples useful just cite the book 'Anisotropy of Metamaterials', and make reference to this repository.

**License**: This code is published on Apache Licence, so you can use it as you want. We hope you enjoy it and cite us if you find this code useful.

**Feedback**: In case of questions, comments or suggestions just contact me. I am not technical support staff, however, I am always glad to discuss with open-minded people.




