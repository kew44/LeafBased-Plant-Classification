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

	imagesListingDF = pandas.read_csv(imagesListingFile, header=0)  # First line of data is taken as the column headings

	labImagesListingDF = imagesListingDF[imagesListingDF.source == "lab"]  # subset of images that are lab images
	fieldImagesListingDF = imagesListingDF[imagesListingDF.source == "field"]  # subset of images that are field images

	imagesCount = len(imagesListingDF)
	labImagesCount = len(labImagesListingDF)
	fieldImagesCount = len(fieldImagesListingDF)


	print("The Leafsnap dataset's Images Listing file has been read in as a DataFrame structure")
	print("\nInformation about the Leafsnap Dataset:")
	print("\tTotal number of images:\t", imagesCount)
	print("\tNumber of lab images:\t", labImagesCount)
	print("\tNumber of field images:\t", fieldImagesCount)

	# string1 = labImagesListingDF[1:2].get('source')
	#print(labImagesListingDF[1:3].values)

	for imageNum in range(1, 2):
		imagePath = getPropertyValue(imagesListingDF, imageNum, IMAGE_PATH)
		imageFullPath = leafsnapDirectory + imagePath

		#image = ImageProcessing.openImage(imageFullPath)
		#image = ImageProcessing.openImage("data/leafsnap-dataset/dataset/images/lab/maclura_pomifera/pi2235-01-1-828.jpg")
		#image = ImageProcessing.openImage("data/leafsnap-dataset/dataset/images/lab/aesculus_hippocastamon/ny1016-05-4.jpg")
		image = ImageProcessing.openImage("data/leafsnap-dataset/dataset/images/lab/ulmus_glabra/ny1074-09-2.jpg")

		print(image)
		#ImageProcessing.displayImage(imageFullPath, image)


		imageHeight = image.shape[0]  # number of rows in the 2D array that represents the image
		imageWidth = image.shape[1]  # number of columns in the 2D array that represents the image

		imageSource = getPropertyValue(imagesListingDF, imageNum, SOURCE)

		if imageSource == "lab":
			# Crop image to size and colour patch rulers on the right and bottom
			image = image[0:imageHeight-120, 0: imageWidth-190]

		ImageProcessing.displayImage(imageFullPath, image)

		image = ImageProcessing.enhanceImage(image)

		ImageProcessing.displayImage(imageFullPath, image)

		imageHistogram = cv2.calcHist(image, [0], None, [256], [0,256])


		pyplot.plot(imageHistogram, color='g')
		pyplot.xlim([0, 256])
		pyplot.show()


		thresholdValue, image = ImageProcessing.segmentImage(image)

		print("Thresholded Image:")
		print(image)
		ImageProcessing.displayImage(imageFullPath, image)

		image = ImageProcessing.morphImage(image)
		ImageProcessing.displayImage(imageFullPath, image)

		#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		#imageG = cv2.imread("data/leafsnap-dataset/dataset/images/lab/acer_palmatum/wb1129-04-4.jpg", cv2.IMREAD_GRAYSCALE)



		#print(len(image))
		#print(len(image[0]))
		#print(len(image[0][0]))
		#ImageProcessing.displayImage(imageFullPath, image)



	"""2. Image Preprocessing"""


	"""3. Obtain Features"""

	"""
		The Features being used are:
			Hu's 7 invariant moments
			Haralick's 14 Texture Descriptors
	"""
	features =  [
					"Hu1", "Hu2", "Hu3", "Hu4", "Hu5", "Hu6", "Hu7", "TD1", "TD2", "TD3", "TD4", "TD5", "TD6", "TD7",
					"TD8", "TD9", "TD10", "TD11", "TD12", "TD13", "TD14"
				]

	"""
		featureMatrixDF is a list of the feature vectors for all the images as a DataFrame structure
		Each row number corresponds to the row number in the imagesListingDF that is being used, i.e. row i in
		featureMatrixDF is the feature vector of the image at row i in imagesListingDF 
		
	"""
	featureMatrixDF = pandas.DataFrame(columns=features)

	ImageProcessing.getImageFeatures(image)



	"""4. Training"""





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



main()