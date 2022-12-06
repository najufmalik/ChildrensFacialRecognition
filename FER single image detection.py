# Reinstall FER incase there are any recent updates
pip install FER

# Import FER and use plt
from fer import FER
import matplotlib.pyplot as plt 
%matplotlib inline

# Find one image to capture emotions
test_image_one = plt.imread("\surprise\A1.png")
emo_detector = FER(mtcnn=True)

# Get the emotions found from the test image
captured_emotions = emo_detector.detect_emotions(test_image_one)

# Print out the emotions along with the image
print(captured_emotions)
plt.imshow(test_image_one)

# Use top_Emotion() function for the dominant emotion
dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
print(dominant_emotion, emotion_score)
