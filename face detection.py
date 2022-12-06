from fer import FER
import cv2

emotion_detector = FER(mtcnn=True)

test_img = cv2.imread("fearVSsurprise(2x2).png")
analysis = emotion_detector.detect_emotions(test_img)

# Get analysis of the test image including multiple faces
print(analysis)
