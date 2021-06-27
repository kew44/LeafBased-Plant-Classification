"""
	COMP702 (Image Processing and Computer Vision) 2021
						PROJECT

				Leaf-Based Plant Classification

			Developed by: Talha Vawda (218023210)


		This project has been developed using:
			Python 3.8.1
			PyCharm 2019.3.3 (Professional Edition) Build #PY-193.6494.30


	Acknowledgements:
		1.
"""
import pandas
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

import ImageProcessing # My file

from IPython.display import display


# The column names of the Images Listing file with their indexes (in the DataFrame) as CONSTANTS
FILE_ID = 0
IMAGE_PATH = 1
SEGMENTED_PATH = 2
SPECIES = 3
SOURCE = 4

# The column names in a array
COLUMNS = ["file_id", "image_path", "segmented_path", "species", "source"]


def main():
	"""1. Obtain Dataset"""

	"""
		Using the Leafsnap dataset

		The leafsnap-dataset-images.csv file contains a listing of all the images

		Using pandas library to read in this file as a DataFrame structure
	"""
	leafsnapDirectory = "data/leafsnap-dataset/"
	imagesListingFile = leafsnapDirectory + "leafsnap-dataset-images.csv"

	# DataFrame of all the images
	allImagesListingDF = pandas.read_csv(imagesListingFile, header=0)  # First line of data is taken as the column headings

	labImagesListingDF = allImagesListingDF[allImagesListingDF.source == "lab"]  # subset of images that are lab images
	fieldImagesListingDF = allImagesListingDF[allImagesListingDF.source == "field"]  # subset of images that are field images

	imagesCount = len(allImagesListingDF)
	labImagesCount = len(labImagesListingDF)
	fieldImagesCount = len(fieldImagesListingDF)


	print("The Leafsnap dataset's Images Listing file has been read in as a DataFrame structure")
	print("\nInformation about the Leafsnap Dataset:")
	print("\tTotal number of images:\t", imagesCount)
	print("\tNumber of lab images:\t", labImagesCount)
	print("\tNumber of field images:\t", fieldImagesCount)

	# string1 = labImagesListingDF[1:2].get('source')
	#print(labImagesListingDF[1:3].values)



	"""2. Image Preprocessing and Obtaining Features"""

	# The DataFrame of the images we are going to be using
	imagesListingDF = labImagesListingDF # Using the lab images



	"""
		featureMatrixDF is a list of the feature vectors for all the images as a DataFrame structure
		Each row number corresponds to the row number in the imagesListingDF that is being used, i.e. row i in
		featureMatrixDF is the feature vector of the image at row i in imagesListingDF 

	"""
	featureMatrixDF = pandas.DataFrame(columns=ImageProcessing.FEATURES)

	for imageNum in range(len(imagesListingDF)):
	#for imageNum in range(0, 2):
		imagePath = getPropertyValue(imagesListingDF, imageNum, IMAGE_PATH)
		imageFullPath = leafsnapDirectory + imagePath

		print(imageNum, end=":\t")

		image = ImageProcessing.openImage(imageFullPath)

		#Testing different images
		#image = ImageProcessing.openImage("data/leafsnap-dataset/dataset/images/lab/maclura_pomifera/pi2235-01-1.jpg")
		#image = ImageProcessing.openImage("data/leafsnap-dataset/dataset/images/lab/aesculus_hippocastamon/ny1016-05-4.jpg")
		#image = ImageProcessing.openImage("data/leafsnap-dataset/dataset/images/lab/ulmus_glabra/ny1074-09-2.jpg")
		#image = ImageProcessing.openImage("data/leafsnap-dataset/dataset/images/lab/ulmus_glabra/ny1074-07-4.jpg")

		#print(image)
		#ImageProcessing.displayImage(imageFullPath, image)


		imageHeight = image.shape[0]  # number of rows in the 2D array that represents the image
		imageWidth = image.shape[1]  # number of columns in the 2D array that represents the image

		imageSource = getPropertyValue(imagesListingDF, imageNum, SOURCE)

		if imageSource == "lab":
			"""
				Crop image to size and colour patch rulers on the right and bottom
				Since x-axis increases downwards and y-axis increases rightwards, we start from 0 on both coordinates and crop at the end
			"""
			image = image[0:imageHeight-120, 0: imageWidth-190]

		#ImageProcessing.displayImage(imageFullPath, image)

		enhancedImage = ImageProcessing.enhanceImage(image)

		#ImageProcessing.displayImage(imageFullPath, enhancedImage)


		#imageHistogram = cv2.calcHist(enhancedImage, [0], None, [256], [0, 256])

		#pyplot.plot(imageHistogram, color='g')
		#pyplot.xlim([0, 256])
		#pyplot.show()


		enhancedImage2 = ImageProcessing.gammaTransformImage(enhancedImage)

		#ImageProcessing.displayImage(imageFullPath, enhancedImage2)


		thresholdValue, segmentedImage = ImageProcessing.segmentImage(enhancedImage2)

		#print("Thresholded Image:")
		#print(segmentedImage)
		#ImageProcessing.displayImage(imageFullPath, segmentedImage)

		morphedImage = ImageProcessing.morphImage(segmentedImage)
		#ImageProcessing.displayImage(imageFullPath, morphedImage)

		# Our greyscale image is the enhancedImage and our binary thresholded image is the morphedImage
		featureVector = ImageProcessing.getImageFeatures(enhancedImage, morphedImage)

		featureMatrixDF.loc[imageNum] = featureVector

		# image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		# imageG = cv2.imread("data/leafsnap-dataset/dataset/images/lab/acer_palmatum/wb1129-04-4.jpg", cv2.IMREAD_GRAYSCALE)

		# print(len(image))
		# print(len(image[0]))
		# print(len(image[0][0]))
		# ImageProcessing.displayImage(imageFullPath, image)


	"""4. Classification - Training and Testing"""

	# The label (Classification) of an image (being represented by its feature vector) is the species of that image
	labelsDF = imagesListingDF["species"]

	"""
		Splitting the labelled dataset into a Training Set (67%) and a Test Set (33%) and doing the training and testing
		Setting train_size to 0.67 and test_size will be automatically set to 0.33 (1.0-0.67)
		
		The data is shuffled before splitting (by default). However, we are specifying the random_state value (this 
		fixes the seed of the pseudorandom number generator)
		so that the dataset is shuffled the same way on each run, allowing us to accurately compare different ML algorithms 

		featuresTest is what we are going to use to predict the species to test our model
		labelsTest matrix is the 'ground truth' labels (i.e. correct species)
		
		Since there are many different species, it would be preferable to maintain the splitting within each species
		as well, i.e. the images for each species are split such that 67% contributes towards the Training Set and 33%
		contributes towards the Testing Set. If the splitting is only done across the entire dataset then we will end 
		up with a scenario where many or all images of a species are selected for the Training Set and none (or very) 
		litte are selected for the Testing Set, and vice versa, which would reduce the accuracy of the model.
		I.e. We want to ensure that relative class frequencies is approximately preserved in each train and test fold

		Since we are not specifying a random_state value, the train and test datasets will be shuffled differently on every run
			If an integer value is specified, a same/similar shuffle will be done each time resulting in the same train and test datasets
	"""

	print()
	print("The species along with their number of occurrences in the dataset we're using:")

	speciesDict = Counter(labelsDF)
	for key in speciesDict:
		# format the printing so that the numbers are directly beneath each other (species have variable name length)
		print("\t", '{species: <30}'.format(species=key), speciesDict[key])

	print()

	featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(featureMatrixDF, labelsDF, train_size=0.67, stratify=labelsDF, random_state=1)

	print("Splitting sizes:")
	print("\tfeaturesTrain:\t", featuresTrain.shape)
	print("\tfeaturesTest:\t", featuresTest.shape)
	print("\tlabelsTrain:\t", labelsTrain.shape)
	print("\tlabelsTest:\t", labelsTest.shape)

	print()
	# Verify that Stratified splitting was done
	print("Verifying Stratification: (Compare with above listing)")
	speciesTrainingDict = Counter(labelsTrain)
	speciesList = []

	for key in speciesTrainingDict:
		speciesList.append(key)
		# format the printing so that the numbers are directly beneath each other (species have variable name length)
		print("\t", '{species: <30}'.format(species=key), speciesTrainingDict[key])


	print()

	"""Classifiers"""
	#Multi-Layer Perceptron
	#‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
	# Our dataset is sufficiently large, so we're sticking with the default 'adam' solver
	mlpClassifier = MLPClassifier(random_state=1, max_iter=1000)

	from sklearn.linear_model import LogisticRegression
	lrClassifier = LogisticRegression(max_iter=1000)

	from sklearn.svm import SVC
	svcClassifier = SVC(random_state=1)

	from sklearn.svm import LinearSVC
	lsvcClassifier = LinearSVC(random_state=1)

	from sklearn.naive_bayes import GaussianNB
	gnbClassifier = GaussianNB()

	from sklearn.linear_model import Perceptron
	pClassifer = Perceptron(random_state=1)

	from sklearn.linear_model import PassiveAggressiveClassifier
	paClassifer = PassiveAggressiveClassifier(random_state=1)

	from sklearn.neighbors import KNeighborsClassifier
	knClassifier = KNeighborsClassifier() # Default: k=5, distance used is Euclidean Distance, weights = uniform (all points in neighbourhood weighted equally)

	from sklearn.ensemble import RandomForestClassifier
	rfClassifier = RandomForestClassifier(random_state=1)


	for classifier in [mlpClassifier, lrClassifier, svcClassifier, lsvcClassifier, gnbClassifier, pClassifer, paClassifer, knClassifier, rfClassifier]:
		classifier.fit(featuresTrain, labelsTrain)
		labelPredictions = classifier.predict(featuresTest)

		print("Classifier:", classifier.__class__.__name__)

		"""
			normalize = True -> returns fraction (in decimal) of correctly classified samples (best performance = 1)
			normalise = False -> returns count  of correctly classified samples
			default is  normalise = True
		"""
		accuracy = metrics.accuracy_score(labelsTest, labelPredictions)
		print("Accuracy: ", round(accuracy, 3)) # Round to 3 decimal places when displaying

		# 'labels parameter by default uses the labels passed in by the ground truth (labelsTest)'
		# average='micro' since we are doing Single-Label Classification - the other options macro/weighted/samples are for Multi-Label Classification
		# Return 0 if there is any division by 0 about to take place

		precision = metrics.precision_score(labelsTest, labelPredictions, average='micro', zero_division=0)
		print("Precision: ", round(precision, 3))

		recall = metrics.recall_score(labelsTest, labelPredictions, average='micro', zero_division=0)
		print("Recall: ", round(recall, 3))

		f1 = metrics.f1_score(labelsTest, labelPredictions, average='micro', zero_division=0)
		print("F1 Score: ", round(f1, 3))

		jaccard = metrics.jaccard_score(labelsTest, labelPredictions, average='micro')
		print("Jaccard Score: ", round(jaccard, 3))

		hammingLoss = metrics.hamming_loss(labelsTest, labelPredictions)
		print("Hamming Loss: ", round(hammingLoss, 3))

		print()
		print("Confusion Matrix: ", metrics.confusion_matrix(labelsTest, labelPredictions))
		print()
		print("Classification Report showing the main Classification metrics for each label (species):\n", metrics.classification_report(labelsTest, labelPredictions, target_names=speciesList, zero_division=0))
		print("-----------------------------------")













def printImagesListing(imagesListingDF):
	"""
		Iterate through the given DataFrame representing images from the images listing file and display the values
		of the DataFrame

		Need to write my own function to display since the size of the  values in the dataframe (specificially the
		file paths are long and get truncated when using fucntions to display using the dataframe itself)

		:param imagesListingDF:     A DataFrame representing a subset of the Leafsnap dataset's images listing file.
									The columns of the DataFrame are 'file_id', 'image_path', 'segmented_path', 'species', 'source'
		:return:                    None
	"""

	# Column Headings
	print("file_id\t\timage_path\t\t\t\t\t\t\t\t\t\t\tsegmented_path\t\t\t\t\t\t\t\t\t\t\tspecies\t\t\tsource")

	print()

	for row, values in imagesListingDF.iterrows():
		print(values["file_id"], "\t\t", values["image_path"], "\t", values["segmented_path"], "\t", values["species"], "\t", values["source"], sep="")


def getImageListing(imagesListingDF, row: int):
	"""
		:param imagesListingDF:  A DataFrame representing a subset of the Leafsnap dataset's images listing file.
		:param row:   The row number in the DataFrame indicating the image
		:return: An array of the values of the image at row
	"""
	image = []

	for column in range(len(COLUMNS)):
		print(imagesListingDF.iloc[row, column])

	return image


def getPropertyValue(imagesListingDF, row: int, property: int):
	"""
		:param imagesListingDF:  A DataFrame representing a subset of the Leafsnap dataset's images listing file.
		:param row:   The row number in the DataFrame indicating the image
		:param property: The property (column) which we want to get the value of for this image at the specified row.
				property is one of FILE_ID = 0, IMAGE_PATH = 1, SEGMENTED_PATH = 2, SPECIES = 3, SOURCE = 4
		:return: The value of the property of the image specified at the row number
	"""
	if property in [FILE_ID, IMAGE_PATH, SEGMENTED_PATH, SPECIES, SOURCE]:
		return imagesListingDF.iloc[row, property]
	else:
		return imagesListingDF.iloc[row, FILE_ID] # Default to return if invalid property given


# Run the main method if this python file is being executed/run directly (either from IDE or Command Line)
if __name__ == '__main__':
	main()