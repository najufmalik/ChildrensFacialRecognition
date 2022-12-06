from deepface import DeepFace

face_analysis = DeepFace.analyze(img_path = "A1.PNG")

# Get the results from the DeepFace library on the analyzed image
print(face_analysis)
