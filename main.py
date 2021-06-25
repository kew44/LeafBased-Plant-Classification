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
import  pandas
import ImageProcessing # My file

from IPython.display import display


# The column names of the Images Listing file
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
	# print(labImagesListingDF[1:3].values)

	for image in range(5):
		imagePath = getImagePath(labImagesListingDF, image)
		imageFullPath = leafsnapDirectory + imagePath
		image = ImageProcessing.openImage(imageFullPath)
		#ImageProcessing.displayImage(imageFullPath, image)



	"""2. Image Preprocessing"""



	"""3. Training"""





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

	return  image


def getImagePath(imagesListingDF, row: int):
	"""
		:param imagesListingDF:  A DataFrame representing a subset of the Leafsnap dataset's images listing file.
		:param row:   The row number in the DataFrame indicating the image
		:return: The image_path of the image at the row number
	"""
	return imagesListingDF.iloc[row, 1] # COLUMNS[1] = "image_path"


main()