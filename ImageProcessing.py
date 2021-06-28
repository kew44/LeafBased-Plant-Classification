import cv2
import numpy
import math
import mahotas

#CONSTANTS
MEDIAN = 0
GAUSSIAN = 1

"""
	The Features being used are:
		Area, Perimeter of Segmented image
		Mean Intensity of Grayscale Image 
		Hu's 7 invariant moments
		The following of Haralick's 14 Texture Descriptors/Features: Energy, Contrast, Correlation, Homogeneity, Entropy
"""
FEATURES = [
			"Area", "Perimeter", "Mean Intensity", "Hu1", "Hu2", "Hu3", "Hu4", "Hu5", "Hu6", "Hu7",
			"Energy", "Contrast", "Correlation", "Homogeneity", "Entropy"
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


	#Doing Smoothing

	if filterMethod == MEDIAN:
		enhancedImage = cv2.medianBlur(image, 5, 0)
	elif filterMethod == GAUSSIAN:
		enhancedImage = cv2.GaussianBlur(image, (5, 5), 0)
	else: # Default for invalid filterMethod given - Do Gaussian
		enhancedImage = cv2.GaussianBlur(image, (5, 5), 0)

	return enhancedImage


def gammaTransformImage(image):
	"""
		Apply Gamma Transformation to the image
		Only do the Gamma Transformation if the leaf is small in size

		:param image:
		:return:
	"""

	#if numpy.sum(image > 150) < 5000:
		# Gamma Transformation
	gtImage = numpy.array(image ** 0.5, dtype='uint8')
	#else:
		#gtImage = image


	return gtImage



def segmentImage(image):
	"""

		:param image: A grayscale image
		:return: thresholding image (black pixels indicate the image [leaf])
	"""
	# Do Thresholding

	thresholdValue = 128 # Set threshold value, being the middle greyscale intensity level
	maxIntensity = 255

	"""
		Since thresholding results in a binary image of black and white pixels with the black pixels being represented
		by a 0 value and the white pixels being represented by a 1 value and Image Moments look at pixel intensity of 
		the image, we want the foreground (Region of Interest) to be represented by white pixels and the background 
		to be represented by black pixels. Since for our dataset, the foreground (the leaves) is darker than the background
		in the images, we want to do Binary Thresholding but inversing the intensities (if pixelIntensity <= Threshold then 
		assign value 1, else assign value 0)
		[Note: The threshold() function assigns 255 instead of 2. So intensity values are either 0 or 255]
	"""
	T, segmentedImage = cv2.threshold(image, thresholdValue, maxIntensity, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # Global thresholding

	#[DEPRECATED - Doing Gamma Transform instead (there are some very large leaves that take up most of the segmented image so not doing this adaptative approach anymore)]
	#if Otsu's method didn't work properly (too many white pixels) then use adaptative thresholding
	#countWhitePixels = numpy.sum(segmentedImage==maxIntensity)

	#if countWhitePixels > (image.shape[0]*image.shape[1] / 3):
	#	segmentedImage = cv2.adaptiveThreshold(image, maxIntensity, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV , 5, 2) # block size of 5
	#	T = -1 # since adaptative thresholding was used, there's no single threshold value


	#print("Threshold value: ", T)

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
		The Features being used are:
			Area, Perimeter of Segmented image
			Mean Intensity of Grayscale Image
			Hu's 7 invariant moments
			The following of Haralick's 14 Texture Descriptors/Features: Energy, Contrast, Correlation, Homogeneity, Entropy

		:param grayscaleImage: the image (as a 2D array of grey-level intensities)
		:param binaryImage: the image after thresholding was done
		:return: The feature vector (as a list) of this image
	"""

	featureVector = []

	"""
			Since a pixel is a 1x1 square block, the area of a leaf is the number of pixels (white pixels due to segmentation) 
			that make up the leaf in the image
		"""
	area = numpy.sum(binaryImage == 255)
	featureVector.append(area)

	"""
		Perimeter

		- Get contours of image

		cv2.RETR_TREE - retrieves all of the contours and reconstructs a full hierarchy of nested contours.
		cv2.RETR_EXTERNAL - retrieves only the extreme outer contours
			-> Doing this method since we are categorising leaves by their shape and there are no
				holes in the leaves
			-> Thus since we're only retrieving the extreme outer contour, there is only one element in contours

		The boundary is not straight lines (as is the case with a square or rectange) so we're storing
		all points of the contour -> cv2.CHAIN_APPROX_NONE
	"""
	contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	if len(contours) > 0:
		perimeter = cv2.arcLength(contours[0], True)
	else:
		perimeter = 0

	featureVector.append(round(perimeter, 4))

	meanIntensity = cv2.mean(grayscaleImage)
	featureVector.append(round(meanIntensity[0], 4)) # Since image is grayscale, the mean value is the first channel


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
		featureVector.append(round(m, 4))


	"""
		Haralick's Textural Features
		
		Energy/Uniformity (Angular Second Moment) is Feature Number 1 (index = 0)
		Contrast is Feature Number 2 (index = 1)
		Correlation is Feature Number 3 (index = 2)
		Homogeneity (Inverse Difference Moment) is Feature Number 5 (index = 4)
		Entropy is Feature Number 9 (index = 8)
		
		distance default is 1
		We are taking the average/mean of each feature in all four angle directions (0, 45, 90, 135 degrees) - This 
		results in rotational invariance
		
		
	"""
	haralickTFeatures = mahotas.features.haralick(grayscaleImage, return_mean=True)
	featureVector.append(round(haralickTFeatures[0], 4))  # Energy
	featureVector.append(round(haralickTFeatures[1], 4))  # Contrast
	featureVector.append(round(haralickTFeatures[2], 4))  # Correlation
	featureVector.append(round(haralickTFeatures[4], 4))  # Homogeneity
	featureVector.append(round(haralickTFeatures[8], 4))  # Entropy

	print("\t\tFeature Vector:\t", featureVector, end="\n\n")

	return featureVector
