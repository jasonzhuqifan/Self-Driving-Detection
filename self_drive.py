from ObjectDetection import ObjectDetection
from LaneDetection import LaneDetection
import cv2
import argparse
import time
from multiprocessing import Process, Manager, Queue, Pool

def main():
	print "Loading modules..."
	objectDetection = ObjectDetection()
	laneDetection = LaneDetection()
	print "Loading done..."

	multProcManager = Manager()
	videoQueue = multProcManager.Queue() # Raw images are pushed to a queue waiting to be processed
	videocap_proc = Process(target = videoToFrames, args = (videoQueue,))
	videocap_proc.start()
	starttime = time.time()
	
	while videoQueue.qsize() == 0:
		continue
	image = videoQueue.get()
	image = cv2.resize(image,(1280, 740))
	writer = cv2.VideoWriter('detection.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (image.shape[1],image.shape[0]), True)
	while True:
		if videoQueue.qsize() > 0:
			starttime = time.time()
			image = videoQueue.get()
			image = cv2.resize(image,(1280, 740))
			finalImage = run(objectDetection,laneDetection, image)
			#finalImage = cv2.resize(image,(1280, 740))
			cv2.imshow("legend", finalImage)
			writer.write(finalImage)
			if cv2.waitKey(20) & 0xFF == ord('q'):  # Hit 'q' to quit anytime
				break
		elif time.time() - starttime >= 0.8:
			break
	cv2.destroyAllWindows()
	writer.release()
	videocap_proc.terminate()
	videocap_proc.join()

def run(objectDetection, laneDetection, img):
	newimg = objectDetection.runDetection(img)
	linedImage = laneDetection.runDetection(img)
	finalImage = cv2.addWeighted(newimg, 0.8, linedImage, 1.0, 0.0)
	return finalImage


def videoToFrames(queue):
	video = cv2.VideoCapture('./video2.mp4')
	while(video.isOpened()):
		check, frame = video.read()
		if check:
			queue.put(frame)
		else:
			break
	video.release()

if __name__ == '__main__':
    main()