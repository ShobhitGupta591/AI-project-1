import cv2

#car image
img_file="car Image.jpeg"

# video_file="Tesla Dashcam.mp4"  #video
video_file = "Pedestrians Compilation.mp4"

#pre trained car detector classifier
car_classifier_file="car_detector.xml"

#pre trained car detector classifier
human_classifier_file="human_detector.xml"

#classifiers
car_tracker = cv2.CascadeClassifier(car_classifier_file)
human_tracker = cv2.CascadeClassifier(human_classifier_file)


#create openCV image
img=cv2.imread(img_file)
video = cv2.VideoCapture(video_file )


#continue untill the video ends
while True:

    #read the current frame
    (read_successful,frame)=video.read()
    # here frame is equivalent to img above, basically its a cv2.imread() object

    # convert to gray frame
    if read_successful:
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break
    

    cars = car_tracker.detectMultiScale(gray_frame)
    humans = human_tracker.detectMultiScale(gray_frame)


    for (x,y,h,w) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
    for(x,y,h,w) in humans:
        cv2.rectangle(frame, (x,y), (x+h, y+w), (0,255,255), 2)
    cv2.imshow("Car Detector", frame)
    key = cv2.waitKey(1)

    if key==83 or key==113:
        break

video.release()










#explaination code

'''
#convert to black n white
black_n_white = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)






#car classifier
car_tracker = cv2.CascadeClassifier(car_classifier_file)


#detect car
cars = car_tracker.detectMultiScale(black_n_white)

for (x,y,h,w) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

print(cars)




#display the image with the car spotted
# cv2.imshow("clever program car detector", black_n_white)
cv2.imshow("clever program car detector", img)

#dont auto close
cv2.waitKey()


print("code completed")
'''