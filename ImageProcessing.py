import cv2

def openImage(imageName):
	"""
		Read an image from the specified path
		:param imageName: The name of the image, include its directory path and file extension
		:return: The image in the form of an intensity values matrix
	"""

	"""
		imread(fiePath, flag)
			flag - how the image should be read
				- cv2.IMREAD_COLOR (default) - transparency is neglected
				- cv2.IMREAD_GRAYSCALE 
				- cv2.IMREAD_UNCHANGED  

		If the image can't be read then an empty matrix is returned (or is an error given?)
			
	"""
	image = cv2.imread(imageName, cv2.IMREAD_COLOR)
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
