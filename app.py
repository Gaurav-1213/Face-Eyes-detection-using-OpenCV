#from typing_extensions import runtime
from flask import Flask,render_template,Response, request
import cv2


app = Flask(__name__)

camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)

def gen_frames():
    while True:
        success,frame = camera.read()   # read camera frames
        if not success:
            break   # in case camera is faulty
        else:
            detector = cv2.CascadeClassifier('haar-cascade-files/haarcascade_frontalface_default.xml')  # creating object of class to extract methods (of class) throuh that object
            eye_cascade = cv2.CascadeClassifier('haar-cascade-files/haarcascade_eye.xml')
            faces = detector.detectMultiScale(frame,1.1,7)  # detect faces from it
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   # convert color frame into gray scale
            
            # draw rectangle around each face
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)  # (255,0,0) = (R,G,B)
                roi_gray= gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale( roi_gray, 1.1,3)
                for (ex,ey,ew,eh) in eyes:          # in face for loop/ in face rectangle detect Eyes
                    cv2.rectangle( roi_color, (ey,ex), (ex+ew, ey+eh), (0,255,0), 2)   # and draw rectangle on eyes too.
            
            ret,buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='mutipart/x-mixed-replace; boudary = frame')
    

if __name__ =="__main__":
    app.run(debug =True)