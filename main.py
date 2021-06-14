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


"""1. Obtain Dataset"""


"""
	Using the Leafsnap dataset
	
	The leafsnap-dataset-images.csv file contains a listing of all the images
	
	Using pandas library to read in this file as a DataFrame structure
"""
leafsnapDirectory = "data/leafsnap-dataset/"
imagesListingFile = leafsnapDirectory + "leafsnap-dataset-images.csv"

imagesListingDF = pandas.read_csv(imagesListingFile, header=0) #First line of data is taken as the column headings

labImagesListingDF = imagesListingDF[imagesListingDF.source=="lab"] #subset of images that are lab images
fieldImagesListingDF = imagesListingDF[imagesListingDF.source=="field"] #subset of images that are field images

imagesCount = len(imagesListingDF)
labImagesCount = len(labImagesListingDF)
fieldImagesCount = len(fieldImagesListingDF)


print("The Images Listing file has been read in as a DataFrame structure")
print("\nInformation about the Leafsnap Dataset:")
print("\tTotal number of images:\t", imagesCount)
print("\tNumber of lab images:\t", labImagesCount)
print("\tNumber of field images:\t", fieldImagesCount)



"""2. Image Preprocessing"""



"""3. Training"""