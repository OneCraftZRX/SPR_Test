import pytesseract
import cv2
import numpy
import easyocr


cap = cv2.VideoCapture(0)  
#reader = easyocr.Reader(['ch_sim','en'], gpu = True) # need to run only once to load model into memory

while True:
    
    ret, frame = cap.read() 
    cv2.imshow('frame', frame)  
    #result = reader.readtext(frame, allowlist ='0123456789',detail=0,min_size=100)
    #print(result)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  

cap.release()  
cv2.destroyAllWindows()