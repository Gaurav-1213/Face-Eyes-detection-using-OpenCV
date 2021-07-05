from flask import Flask,render_template,Response
import cv2

app=Flask(__name__)
camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            detector = cv2.CascadeClassifier('haar-cascade-files/haarcascade_frontalface_default.xml')  # creating object of class to extract methods (of class) throuh that object
            eye_cascade = cv2.CascadeClassifier('haar-cascade-files/haarcascade_eye2.xml')
            #smile_cascade= cv2.CascadeClassifier('haar-cascade-files/haarcascade_smile.xml')
            faces = detector.detectMultiScale(frame)  # detect faces from it
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   # convert color frame into gray scale
            
            # draw rectangle around each face
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)  # .rectang(frame, startingPoint, endpoint, color, thickness) (255,0,0) = (B,G,R)
                roi_gray= gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale( roi_gray)
                #smile = smile_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:          # in face for loop/ in face rectangle detect Eyes
                    cv2.rectangle( roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)   # , 2) is thickness of eye boxes, and draw rectangle on eyes too.
                #for (sx,sy,sw,sh) in smile:          # in face for loop/ in face rectangle detect Eyes
                    #cv2.rectangle( roi_color, (sx, sy), (sx+sw, sy+sh), (0,255,0), 2)   # , 2) is thickness of eye boxes, and draw rectangle on eyes too.
            
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)