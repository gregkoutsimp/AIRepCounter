import cv2
import PoseModule as pm
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

root = tk.Tk()
root.withdraw()

exercise = simpledialog.askstring(title="Exercise",
                                  prompt="Give the name of the exercise:").lower()
print('Upload the video')

vidpath = filedialog.askopenfilename(title = 'Upload the video')
cap = cv2.VideoCapture(str(vidpath))
detector = pm.poseDetector()

count = 0
dir = 0

while True:
    success, img = cap.read()
    img=cv2.resize(img, (1280,720))

    if exercise == 'bicep curls':
        img, count, dir = detector.bicepCurl(img,count,dir) #WORKING
    elif exercise == 'squat':
        img, count, dir = detector.squat(img, count, dir)  # WORKING
    elif exercise == 'pull ups':
        img, count, dir = detector.pullUps(img, count, dir)  # WORKING
    else :
        messagebox.showinfo("Error", "AI Trainer is only available for Squats, Pull Ups and Bicep Curls, more exercises will be added soon")
        break

    detector.boxCounter(img,count)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
