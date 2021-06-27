import cv2
import numpy
import math

#CONSTANTS
MEDIAN = 0
GAUSSIAN = 1

"""
	The Features being used are:
		Hu's 7 invariant moments
		Haralick's 14 Texture Descriptors
"""
FEATURES = [
			"Hu1", "Hu2", "Hu3", "Hu4", "Hu5", "Hu6", "Hu7", "TD1", "TD2", "TD3", "TD4", "TD5", "TD6", "TD7",
			"TD8", "TD9", "TD10", "TD11", "TD12", "TD13", "TD14"
			]


def openImage(imageName):
	"""
		Read an image from the specified path

		For a colour image, the returned image is a 3D array/list - a list of lists of lists.
		The outmost list is represents the list of rows of the image.
		Each element in this outmost list is a list which represents a  row in the image
		This inner list is itself a list of lists - each element in this inner list is a list of size 3
		representing a pixel in the image where the list representing the pixel specifies the RGB values of that
		pixel as [B, G, R]

		For a greysccale image, the returned image is a 2D list where each element at row i and column j represents
		a pixel, with the value being the grey-level intensity of that pixel

		Thus the length of the image variable gives the number of rows in the image (height)
		The length of an inner list gives the number of columns in the image (width)
		The length of an inner inner list is 3 -> the 3 RGB values

		Since colour images increase the complexity of the model, we shall be converting the image to greyscale
		when reading it in


		:param imageName: The name of the image, include its directory path and file extension
		:return: The image in the form of an intensity values matrix
	"""

	"""
		imread(fiePath, flag)
			flag - how the image should be read
				- cv2.IMREAD_COLOR (default) - transparency is neglected
				- cv2.IMREAD_GRAYSCALE 
				- cv2.IMREAD_UNCHANGED  

		If the image can't be read then an empty matrix is returned
			
	"""
	#image = cv2.imread(imageName, cv2.IMREAD_COLOR) #  Read in image as is
	image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)  # Read in image and convert to greyscale
	print("Image '" + imageName + "' has been opened")
	return image


def displayImage(imageName, image):
	"""
		GUI Window to display image on screen
	"""
	windowTitle = "Image Opener - " + imageName

	# cv2.namedWindow(windowTitle, cv2.WINDOW_AUTOSIZE) #this happens by default?
	cv2.imshow(windowTitle, image)

	"""
		Keep image on screen for specified milliseconds
		If 0, then keep on screen till user closes window

		[cv2.waitKey() is a keyboard binding function. Its argument is the time in milliseconds. 
		The function waits for specified milliseconds for any keyboard event. 
		If you press any key in that time, the program continues. 
		If 0 is passed, it waits indefinitely for a key stroke. 
		It can also be set to detect specific key strokes like, if key a is pressed - the function returns which key was pressed]
	"""
	cv2.waitKey(0)



def enhanceImage(image, filterMethod=GAUSSIAN):

	# [Deprecated] Do Histogram Equalisation
	#image = cv2.equalizeHist(image)


	# Gamma Transformation
	image = numpy.array(image ** 0.5, dtype='uint8')

	#Doing Smoothing

	if filterMethod == MEDIAN:
		enhancedImage = cv2.medianBlur(image, 5, 0)
	elif filterMethod == GAUSSIAN:
		enhancedImage = cv2.GaussianBlur(image, (5, 5), 0)
	else: # Default for invalid filterMethod given - Do Gaussian
		enhancedImage = cv2.GaussianBlur(image, (5, 5), 0)

	return enhancedImage




def segmentImage(image):
	"""

		:param image: A grayscale image
		:return: thresholding image (black pixels indicate the image [leaf])
	"""
	# Do Thresholding

	thresholdValue = 128 # Set threshold value, being the middle greyscale intensity level
	maxIntensity = 255
	T, segmentedImage = cv2.threshold(image, thresholdValue, maxIntensity, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Global thresholding

	#if Otsu's method didn't work properly (too many black pixels) then use adaptative thresholding
	countBlackPixels = numpy.sum(segmentedImage==0)

	if countBlackPixels > (image.shape[0]*image.shape[1] / 3):
		segmentedImage = cv2.adaptiveThreshold(image, maxIntensity, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2) # block size of 5
		T = -1 # since adaptative thresholding was used, there's no single threshold value


	print("Threshold value: ", T)

	return T, segmentedImage


def morphImage(image):
	"""
		Do Morphological Image Processing
		:param image:
		:return:
	"""
	se = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 3x3 Structuring Element of a cross type

	morphedImage = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)  # Closing (to fill in gaps/ridges on image edges)

	return morphedImage


def getEdges(image):
	"""

		:param image: An 8-bit grayscale image
		:return:
	"""
	edges = cv2.Canny(image, 0 , 255)


def getImageFeatures(grayscaleImage, binaryImage):
	"""

		:param grayscaleImage: the image (as a 2D array of grey-level intensities)
		:param binaryImage: the image after thresholding was done
		:return: The feature vector (as a list) of this image
	"""
	featureVector = []

	moments = cv2.moments(grayscaleImage, binaryImage=False)
	huMoments = cv2.HuMoments(moments) # Get the 7 Hu Invariant moments as a list

	"""
		Since each Hu Moment in huMoments is a list itself (of a single value),
		flatten the huMoments list to make the list a 1D list (the Hu Moments are now values of this list) 
	"""
	huMoments = huMoments.flatten()

	# Apply log transform to Hu Moments to make them of a comparable scale

	for i in range(len(huMoments)):
		"""
			Since the absolute values of the Hu Moments are small floating point values less than 1,
			the log will return a negative value and multiplied by -1 will return a positive value.
			However if the initial value of the Hu Moment was negative, we want the transformed value
			to also be negative. Thus we multiply by either 1 or -1 (depending on the initial sign of the 
			Hu Moment) to ensure that the sign remains the same
		"""
		huMoments[i] = -1 * math.copysign(1, huMoments[i]) * math.log10(abs(huMoments[i]))


	for m in huMoments:
		featureVector.append(m)


	# The area of a leaf is the number of pixels (black pixels due to segmentation) that make up the leaf in the image
	area = numpy.sum(binaryImage==255)
	featureVector.append(area)

	meanIntensity = cv2.mean(grayscaleImage)


	print(featureVector)

"""==============================================================================================================="""
#Write image to file
#save the image in PNG format in the working directory
#cv2.imwrite(filename, image)



#intensityfile = open("ImageIntensityValues.csv", "w")
#intensityfile.write(imageIntensityMatrix)
#intensityfile.close()


def saveIntensityMatrix(imageName, image):
	"""
	"""

	# I want the name of the file to include the name of the photo
	# self.imageName contains the relative directory, so we need to remove it

	forwardslashIndex = imageName.rfind('/')  # Find the last occurrence of a forward slash in the image filename
	filename = "ImageIntensityValues - '" + imageName[forwardslashIndex + 1:] + "'.csv"


	# Set delimiter so that each pixel row is on the same line
	# fmt (format) - setting it to no decimal places (pixel values are integers)
	# 	- if not specified, values will be saved in scientific format
	#	- See https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html

	numpy.savetxt(filename, image, delimiter=", ", fmt="%.0d")

	print("Intensity Matrix saved to file: " + filename)


def printIntensityMatrix(imageName, image):
	"""
		NOTE:
			The Run console in Pycharm only displays the start and end values of the matrix,
			the entire matrix is not displayed
	"""

	print("Image Pixel Intensity Values of image '" + imageName + "': ")
	print(image)


"""==============================================================================================================="""
class ImageProcessor:
	"""
		A class to represent an image and do processing on that image

		All original images in the Leafsnap dataset are jpg images and all segmented images are png images
		Instance Variables (fields):
			imageName
			image		Variable to store the actual image (its intensity matrix)
	"""

	def __init__(self, imageName):
		"""
			Constructor

			:param imageName: The name of the image, include its directory path and file extension

		"""
		self.imageName = imageName
