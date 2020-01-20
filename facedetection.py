import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img=cv2.imread("photo.jpg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#^^We wanted to have a grayscale image of the pic
faces=face_cascade.detectMultiScale(gray_img,       # "detectMultiScale" >>See at the bottom for explanation
scaleFactor=1.1,                                   #Telling python to search bigger faces by scaling down picture by 
minNeighbors=5)                                     # 1.05= 5% OR 1.5=50% and search again and again til a face is detected
                                                    # Small number means more accuracy but more time

for x, y, w, h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

#^^ x,y= coordinates of upper left corner of the rectangle
# w, h are the width and height. (x+w,x+h) the other 
# corner of the rectangle. 3 is the thickness of rectangle

#print(type(faces))
#print(faces)

resized=cv2.resize(img,(int(img.shape[1]/3),int(img.shape[1]/3)))

#cv2.imshow("Gray",gray_img)          # Show picture in a window with name Gray
cv2.imshow("Gray",resized)
cv2.waitKey(0)                  # Window stays until user presses any key
cv2.destroyAllWindows()


# "detectMultiScale"
# It basically detects coordinates of upperleft corner of the face rectange
# and also gives height and width of the rectangle defining the face 