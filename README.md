# LeafBased-Plant-Classification
 Leaf-Based Plant Classification using the Leafsnap dataset
 

Project Repository Link: https://github.com/talhavawda/LeafBased-Plant-Classification

Download Leafsnap dataset: http://leafsnap.com/static/dataset/leafsnap-dataset.tar

Note that due to its large size, the dataset has been added to .gitignore and is thus not present in this cloud-hosted repository.

## Dependencies

1. Python 3.8

2. The following Python libraries need to be installed on your local pc:
    1. pandas
    2. numpy
    3. OpenCV (cv2)
    4. mahotas
    5. scikit-learn (sklearn)

<br>

## Execution instructions
1. Download the project repository from the GitHub link above and extract the project folder to a location on your pc
2. Download the Leafsnap dataset at the above link (if you do not already have it downloaded)
3. Extract the leafsnap-dataset.tar file using a tool such as 7-Zip
4. Place the 'leafsnap-dataset' folder that was extracted, into the 'data' folder in your local folder (on your pc) of the project repository
    - Verify that the 'leafsnap-dataset' folder has two files ('leafsnap-dataset-readme.txt' and 'leafsnap-dataset-images.txt') and one folder ('dataset') that contains two subfolders ('images' and 'segmented')
5. There is a file called 'leafsnap-dataset-images.csv' in the root folder of the repository. Take this file and move it into the data/leafsnap-dataset folder (this folder contains its corresponding .txt file)
6. To run the program, you can either open the project in an IDE (preferably PyCharm) and run it in the IDE or run the program from the Command Prompt
    - To run the program from the Command Prompt (in Windows):
        - Change directory to the root folder of this project - Navigate to this project folder location on your pc using the 'cd' command
          - The project folder name is 'LeafBased-Plant-Classification'
        - Type in the following command into the Command Prompt: 'python Main.py' (without the inverted commas) to run this project
       
