##########################################
Run HOG-SVM approach
##########################################
The data mentioned in the report should be in a folder "data" which must be one folder level up from "svm_hog.ipynb". In the Data folder don't delete the "FullIJCNN2013" folder.

Tree:
-code_svm
   -svm_hog.ipynb
-data
   -FullIJCNN2013
      - [images]
      - gt.txt

The code produces images and graphs which are stored in the code_svm folder.

##########################################
Run R-CNN approach
##########################################

The data mentioned in the report should be in a subfolder "data" which must be in the same folder as the "network.ipynb". In the data folder don't delete the "FullIJCNN2013" folder.

Tree:
-network.ipynb
-data
   -FullIJCNN2013
      - [images]
      - gt.txt

I also submit a file called "regions_250_09.csv". Copy this file in the "FullIJCNN2013" folder. Since the selective search algorithm takes a while, this file prevents the new computation of the regions.
