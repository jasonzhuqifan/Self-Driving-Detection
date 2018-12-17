from ObjectDetection import ObjectDetection
import cv2

def main():
	print "=== STARTED DOWNLOADING MODEL ==="
	objectDetection = ObjectDetection()
	print "=== FINISHED DOWNLOADING MODEL ==="
	img = cv2.imread('./light.png', 1)
	img = cv2.resize(img,(1280, 740))
	newimg = objectDetection.runDetection(img)

	print(newimg)

	cv2.imshow("legend", newimg)

	cv2.waitKey(0)

	cv2.destroyAllWindows()

if __name__ == '__main__':
    main()