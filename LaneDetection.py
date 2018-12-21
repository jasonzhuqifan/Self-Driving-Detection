import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class LaneDetection:

	def runDetection(self, image):
		#convert to gray image first
		grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#rgbimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#hlsimg = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HLS)
		maskedImg = self.mask(grayImage, image)
		gaussimg = cv2.GaussianBlur(maskedImg,(5,5),0)
		low_thresh = 50
		high_thresh = 150
		# Canny Edge Detection
		cannyedImage = cv2.Canny(gaussimg, low_thresh, high_thresh) 
		# only need region that we interterested in
		croppedImage = self.cropImage(cannyedImage)
		linedImage = self.houghLinesP(croppedImage)
		return linedImage

	def cropImage(self, image):
		height = image.shape[0]
		width = image.shape[1]
		region_of_interest_vertices = [
			(width/10, height*12/13),
			(width/3 - width/7, height/3+height/4),
			(width/3+width/5, height/3+height/4),
			(width - width/10, height*12/13)
		]
		vertices = np.array([region_of_interest_vertices], np.int32)
		mask = np.zeros_like(image)
		match_mask_color = 255
		cv2.fillPoly(mask, vertices, match_mask_color)
		masked_image = cv2.bitwise_and(image, mask)
		return masked_image

	def houghLinesP(self, image):
		lines = cv2.HoughLinesP(
			image,
			rho=3,
			theta=np.pi / 180,
			threshold=50,
			lines=np.array([]),
			minLineLength=50,
			maxLineGap=175
		)
		lineImage = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
		linedImage = self.drawLines(lineImage, lines)
		return linedImage

	def drawLines(self, image, lines, color=[255, 0, 0], thickness=3):

		#image = np.copy(image)
		#print(image)
		line_image = np.zeros(
			(
				image.shape[0],
				image.shape[1],
				3
			),
			dtype=np.uint8,
		)
		# Loop over all lines and draw them on the blank image.
		try:
			for line in lines:
				for x1, y1, x2, y2 in line:
					cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
			# Merge to the original image
		except Exception as e:
			pass
		image = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
		return image

	def mask(self, grayimg, image):
		low_yellow = np.array([20, 120, 80], dtype = "uint8")
		high_yellow = np.array([45, 200, 255], dtype="uint8")
		yellowMask = cv2.inRange(image, low_yellow, high_yellow)
		whiteMask = cv2.inRange(grayimg, 200, 255)
		whiteYelloMask = cv2.bitwise_or(whiteMask, yellowMask)
		maskedImg = cv2.bitwise_and(grayimg, whiteYelloMask)

		return maskedImg





