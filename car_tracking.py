import cv2

''' track car from single image
# our test image file
car_img4 = 'car_4.jpg'

# our pre-trained car classifier file
classifier_file = 'car_detaction.xml'

# load the image in opencv
img = cv2.imread(car_img4)

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# convert our image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# track our cars co-ordinates
cars_coordinates = car_tracker.detectMultiScale(img_gray)
print(cars_coordinates)

# Draw rectangles around the cars
for (x, y, w, h) in cars_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 5)


# display the image
cv2.imshow("Car Detection", img)
cv2.waitKey()
'''

""" Real Time Car Tracking """

# load our video file
video = cv2.VideoCapture("./car video 480p.mp4")
# video = cv2.VideoCapture('video car.mp4')

# our pre-trained car classifier file
car_classifier_file = 'car_detaction.xml'
human_classifier_file = 'haarcascades_fullbody.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(car_classifier_file)
human_tracker = cv2.CascadeClassifier(human_classifier_file)

# Run the loop untill video is end or esc from keyword
while True:
    # Read the current frame
    (read_succefull, frame) = video.read()

    # check for the frame
    if read_succefull:
        # convert the frame into grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        print("Frame not read successfully!")
        break
    
    # detect car in the frame
    cars_coordinates = car_tracker.detectMultiScale(gray_frame)
    humans_coordinates = human_tracker.detectMultiScale(gray_frame)

    # Draw rectangles around the cars
    for (x, y, w, h) in cars_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 5)

    # Draw rectangles around the humans
    for (x, y, w, h) in humans_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 5)

    # display the image
    cv2.imshow("Car Detection", frame)
    if cv2.waitKey(10) == 27:
        break


print("Code Completed!")