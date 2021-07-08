import sys
import urllib.request
from PyQt5.QtWidgets import *
#from PyQt5.QtCore import *
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5 import uic
import numpy as np              #수학적계산을 위함
import cv2                      #OpenCV-face detection
import dlib                     #face landmark detection
from math import atan2, degrees #안경 피팅시 얼굴형에 따라 위치 조정
from sklearn.cluster import KMeans
import glasses_functions


#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("GUI.ui")[0]


class ThreadClass(QtCore.QThread): 
    def __init__(self, parent = None): 
        super(ThreadClass,self).__init__(parent)
    def run(self):
        cam = cv2.VideoCapture(cv2.CAP_DSHOW)
        cam.set(3,1280.)
        cam.set(4,720.)
        while True: 
            ret, frame = cam.read()
            
            cv2.imshow("frame1", frame)
            if not ret:
                break
            k = cv2.waitKey(1)
            if k%256 == 13:
                # Enter pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "face_input.png"
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                
        cam.release()

        cv2.destroyWindow('frame1')


#화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.threadclass = ThreadClass()

        #버튼에 기능을 연결하는 코드
        self.Capture.clicked.connect(self.Capture_Function)
        self.Load_1.clicked.connect(self.Load_Function_1)
        self.Analysis.clicked.connect(self.Analysis_Function)
        self.Load_2.clicked.connect(self.Load_Function_2)
        
        self.Oval.clicked.connect(self.Rectangle_Function)
        self.Round.clicked.connect(self.Square_Function)
        self.Wellington.clicked.connect(self.Triangle_Function)
        self.Square.clicked.connect(self.Round_Function)
        self.Eggplant.clicked.connect(self.Diamond_Function)
        self.Fox.clicked.connect(self.Oval_Function)
        
        self.face_shape.currentIndexChanged.connect(self.comboBoxFunction)
        
    

    def Capture_Function(self) :
        self.threadclass.start()
        

    def Load_Function_1(self):
        self.qPixmapFileVar=QPixmap()
        self.qPixmapFileVar.load("/Users/82103/Final_Project/face_input.png")
        self.qPixmapFileVar = self.qPixmapFileVar.scaled(800,450)
        self.label.setPixmap(self.qPixmapFileVar)

    def comboBoxFunction(self):
        #종류를 두 가지로 추가해도 좋을듯
        if self.face_shape.currentIndex()==0:
            print("얼굴형을 선택하세요")
        
        elif self.face_shape.currentIndex()==1:
            self.face_shape.activated.connect(self.Oval_Function)
            
        elif self.face_shape.currentIndex()==2:
            self.face_shape.activated.connect(self.Round_Function)
            
        elif self.face_shape.currentIndex()==3:
            self.face_shape.activated.connect(self.Rectangle_Function)
            
        elif self.face_shape.currentIndex()==4:
            self.face_shape.activated.connect(self.Square_Function)
            
        elif self.face_shape.currentIndex()==5:
            self.face_shape.activated.connect(self.Diamond_Function)
            
        elif self.face_shape.currentIndex()==6:
            self.face_shape.activated.connect(self.Triangle_Function)

        
        
#############face_shape_analysis#########################################################
        
    def Analysis_Function(self):
        imagepath = "/Users/82103/FInal_Project/face_input.png"
        face_cascade_path = "/Users/82103/Final_Project/haarcascade_frontalface_default.xml"
        predictor_path = "/Users/82103/FInal_Project/shape_predictor_68_face_landmarks.dat"
        faceCascade = cv2.CascadeClassifier(face_cascade_path)
        predictor = dlib.shape_predictor(predictor_path)
        image = cv2.imread(imagepath)
        image = cv2.resize(image, (800, 450))
        original = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray,(3,3), 0)
        faces = faceCascade.detectMultiScale(
            gauss,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(100,100),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        print("found {0} faces!".format(len(faces)) )

        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
            detected_landmarks = predictor(image, dlib_rect).parts()
            landmarks = np.matrix([[p.x,p.y] for p in detected_landmarks])
            
        results = original.copy()


        for (x,y,w,h) in faces:
            cv2.rectangle(results, (x,y), (x+w,y+h), (0,255,0), 2)
            temp = original.copy()
            forehead = temp[y:y+int(0.25*h), x:x+w]
            rows,cols, bands = forehead.shape
            X = forehead.reshape(rows*cols,bands)
            kmeans = KMeans(n_clusters=2,init='k-means++',max_iter=300,n_init=10, random_state=0)
            y_kmeans = kmeans.fit_predict(X)
            
            for i in range(0,rows):
                for j in range(0,cols):
                    if y_kmeans[i*cols+j]==True:
                        forehead[i][j]=[255,255,255]
                    if y_kmeans[i*cols+j]==False:
                        forehead[i][j]=[0,0,0]
            
            forehead_mid = [int(cols/2), int(rows/2) ] #midpoint of forehead
            lef=0 
            pixel_value = forehead[forehead_mid[1],forehead_mid[0] ]
            
            for i in range(0,cols):
                if forehead[forehead_mid[1],forehead_mid[0]-i].all()!=pixel_value.all():
                    lef=forehead_mid[0]-i
                    break
          
            left = [lef,forehead_mid[1]]
            rig=0
            
            for i in range(0,cols):
                if forehead[forehead_mid[1],forehead_mid[0]+i].all()!=pixel_value.all():
                    rig = forehead_mid[0]+i
                    break
           
            right = [rig,forehead_mid[1]]
  
        #line1
        line1 = np.subtract(right+y,left+x)[0]
        #drawing line 2 with circles
        linepointleft = (landmarks[1,0],landmarks[1,1])
        linepointright = (landmarks[15,0],landmarks[15,1])
        line2 = np.subtract(linepointright,linepointleft)[0]
        cv2.line(results, linepointleft,linepointright,color=(0,255,0), thickness = 2)
        cv2.circle(results, linepointleft, 5, color=(255,0,0), thickness=-1)    
        cv2.circle(results, linepointright, 5, color=(255,0,0), thickness=-1)    

        #drawing line 3 with circles
        linepointleft = (landmarks[4,0],landmarks[4,1])
        linepointright = (landmarks[12,0],landmarks[12,1])
        line3 = np.subtract(linepointright,linepointleft)[0]
        cv2.line(results, linepointleft,linepointright,color=(0,255,0), thickness = 2)
        cv2.circle(results, linepointleft, 5, color=(255,0,0), thickness=-1)    
        cv2.circle(results, linepointright, 5, color=(255,0,0), thickness=-1)    

        #drawing line 4 with circles
        linepointbottom = (landmarks[8,0],landmarks[8,1])
        linepointtop = (landmarks[8,0],y)
        line4 = np.subtract(linepointbottom,linepointtop)[1]
        cv2.line(results,linepointtop,linepointbottom,color=(0,255,0), thickness = 2)
        cv2.circle(results, linepointtop, 5, color=(255,0,0), thickness=-1)    
        cv2.circle(results, linepointbottom, 5, color=(255,0,0), thickness=-1)    
        similarity = np.std([line1,line2,line3])
        ovalsimilarity = np.std([line2,line4])

        ax,ay = landmarks[3,0],landmarks[3,1]
        bx,by = landmarks[4,0],landmarks[4,1]
        cx,cy = landmarks[5,0],landmarks[5,1]
        dx,dy = landmarks[6,0],landmarks[6,1]

        import math
        from math import degrees
        
        alpha0 = math.atan2(cy-ay,cx-ax)
        alpha1 = math.atan2(dy-by,dx-bx)
        alpha = alpha1-alpha0
        angle = abs(degrees(alpha))
        angle = 180-angle

        for i in range(1):
          if line3>line1:
            if angle<160:
              print('triangle shape.Forehead is more wider')
              face_shape='triangle '
              break
          if ovalsimilarity<15:
            print('diamond shape. line2 & line4 are similar and line2 is slightly larger')
            face_shape='diamond '
            break
          if line4 > line2:
              if similarity<26:
                  if angle<160:
                      print('squre .Jawlines are more angular')
                      face_shape='squre'
                      break
                  else:
                      print('round .Jawlines are not that angular')
                      face_shape='round'
                      break
              elif angle<160:
                  print('rectangular. face length is long and jawline are angular ')
                  face_shape='rectangular'
                  break
              else:
                  print('oval. face length is long and jawlines are not angular')
                  face_shape='oval'
                  break
          print("Damn! Contact the developer")
        output=results
        cv2.putText(output,face_shape,linepointleft,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0), thickness=2)       
        cv2.imwrite('/Users/82103/Final_Project/shape_detected.png',output)

    def Load_Function_2(self):
        self.qPixmapFileVar=QPixmap()
        self.qPixmapFileVar.load("/Users/82103/Final_Project/shape_detected.png")
        self.qPixmapFileVar = self.qPixmapFileVar.scaled(800,450)
        self.label.setPixmap(self.qPixmapFileVar)

        
################Oval_Face_glasses############################################################
    def Oval_Function(self) :

        imagepath = "/Users/82103/FInal_Project/face_input.png"
        face_cascade_path = "/Users/82103/Final_Project/haarcascade_frontalface_default.xml"
        predictor_path = "/Users/82103/FInal_Project/shape_predictor_68_face_landmarks.dat"
        img_counter=1
        glasses = cv2.imread('/Users/82103/FInal_Project/glasses/fox_1.png', cv2.IMREAD_UNCHANGED)
        
      
        def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
          bg_img = background_img.copy()
          if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
          if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
          b, g, r, a = cv2.split(img_to_overlay_t)   
          mask = cv2.medianBlur(a, 5)
          h, w, _ = img_to_overlay_t.shape
          roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
          img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
          img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)
          bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)
          bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
          return bg_img

        def angle_between(p1, p2):
          xDiff = p2[0] - p1[0]
          yDiff = p2[1] - p1[1]
          return degrees(atan2(yDiff, xDiff))
        
        faceCascade = cv2.CascadeClassifier(face_cascade_path)
        predictor = dlib.shape_predictor(predictor_path)

        image = cv2.imread(imagepath)
        image = cv2.resize(image, (800, 450))
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray,(3,3), 0)
        faces = faceCascade.detectMultiScale(
            gauss,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(100,100),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        print("found {0} faces!".format(len(faces)) )
        
   
        for (x,y,w,h) in faces:
            #cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
         
            pre_detected_landmarks=predictor(image,dlib_rect) 
            detected_landmarks = pre_detected_landmarks.parts()
            landmarks = np.matrix([[p.x,p.y] for p in detected_landmarks])
           
            landmark = image.copy()
            for idx, point in enumerate(landmarks):
                    pos = (point[0,0], point[0,1] )
                    cv2.putText(landmark,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0,0,255) )
                    cv2.circle(landmark, pos, 3, color=(0,255,255))
        
        vec = np.empty([68, 2], dtype = int)   
        for b in range(68):
            vec[b][0] = pre_detected_landmarks.part(b).x 
            vec[b][1] = pre_detected_landmarks.part(b).y 

        ori_img=image.copy()
        result_img=image.copy()

        glasses_center = np.mean([vec[39], vec[42]], axis=0)  
        glasses_size = np.linalg.norm(vec[0] - vec[16])*0.95 #눈의 크기에 따라 안경 사이즈 조절

        angle =-angle_between(vec[39], vec[42])  

        M = cv2.getRotationMatrix2D((glasses.shape[1] / 2, glasses.shape[0] / 2), angle, 1)
        rotated_glasses = cv2.warpAffine(glasses, M, (glasses.shape[1],glasses.shape[0])) 
        result_img = overlay_transparent(result_img, rotated_glasses, glasses_center[0], glasses_center[1], overlay_size=(int(glasses_size), int(glasses.shape[0] * glasses_size / glasses.shape[1])))
        
        cv2.imwrite('/Users/82103/Final_Project/Oval_fitted.png',result_img)

        self.qPixmapFileVar=QPixmap()
        self.qPixmapFileVar.load("/Users/82103/Final_Project/Oval_fitted.png")
        self.qPixmapFileVar = self.qPixmapFileVar.scaled(800,450)
        self.label.setPixmap(self.qPixmapFileVar)

#############Round############################################################################
        

    def Round_Function(self) :

        imagepath = "/Users/82103/FInal_Project/face_input.png"
        face_cascade_path = "/Users/82103/Final_Project/haarcascade_frontalface_default.xml"
        predictor_path = "/Users/82103/FInal_Project/shape_predictor_68_face_landmarks.dat"
        glasses = cv2.imread('/Users/82103/FInal_Project/glasses/Square_1.png', cv2.IMREAD_UNCHANGED)
      
        def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
          bg_img = background_img.copy()
          if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
          if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
          b, g, r, a = cv2.split(img_to_overlay_t)   
          mask = cv2.medianBlur(a, 5)
          h, w, _ = img_to_overlay_t.shape
          roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
          img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
          img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)
          bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)
          bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
          return bg_img

        def angle_between(p1, p2):
          xDiff = p2[0] - p1[0]
          yDiff = p2[1] - p1[1]
          return degrees(atan2(yDiff, xDiff))
        
        faceCascade = cv2.CascadeClassifier(face_cascade_path)
        predictor = dlib.shape_predictor(predictor_path)

        image = cv2.imread(imagepath)
        image = cv2.resize(image, (800, 450))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray,(3,3), 0)
        faces = faceCascade.detectMultiScale(
            gauss,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(100,100),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        print("found {0} faces!".format(len(faces)) )
        
   
        for (x,y,w,h) in faces:
            #cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
         
            pre_detected_landmarks=predictor(image,dlib_rect) 
            detected_landmarks = pre_detected_landmarks.parts()
            landmarks = np.matrix([[p.x,p.y] for p in detected_landmarks])
           
            landmark = image.copy()
            for idx, point in enumerate(landmarks):
                    pos = (point[0,0], point[0,1] )
                    cv2.putText(landmark,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0,0,255) )
                    cv2.circle(landmark, pos, 3, color=(0,255,255))
        
        vec = np.empty([68, 2], dtype = int)   
        for b in range(68):
            vec[b][0] = pre_detected_landmarks.part(b).x 
            vec[b][1] = pre_detected_landmarks.part(b).y 

        ori_img=image.copy()
        result_img=image.copy()

        glasses_center = np.mean([vec[39], vec[42]], axis=0)  
        glasses_size = np.linalg.norm(vec[0] - vec[16])*0.95 #눈의 크기에 따라 안경 사이즈 조절

        angle =-angle_between(vec[39], vec[42])  

        M = cv2.getRotationMatrix2D((glasses.shape[1] / 2, glasses.shape[0] / 2), angle, 1)
        rotated_glasses = cv2.warpAffine(glasses, M, (glasses.shape[1],glasses.shape[0])) 
        try:
            result_img = overlay_transparent(result_img, rotated_glasses, glasses_center[0], glasses_center[1], overlay_size=(int(glasses_size), int(glasses.shape[0] * glasses_size / glasses.shape[1])))
        except:
            print('failed overlay image')

        cv2.imwrite('/Users/82103/Final_Project/Oval_fitted.png',result_img)

        self.qPixmapFileVar=QPixmap()
        self.qPixmapFileVar.load("/Users/82103/Final_Project/Oval_fitted.png")
        self.qPixmapFileVar = self.qPixmapFileVar.scaled(800,450)
        self.label.setPixmap(self.qPixmapFileVar)

#############Rectangle############################################################################
        

    def Rectangle_Function(self) :

        imagepath = "/Users/82103/FInal_Project/face_input.png"
        face_cascade_path = "/Users/82103/Final_Project/haarcascade_frontalface_default.xml"
        predictor_path = "/Users/82103/FInal_Project/shape_predictor_68_face_landmarks.dat"
        glasses = cv2.imread('/Users/82103/FInal_Project/glasses/oval.png', cv2.IMREAD_UNCHANGED)
      
        def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
          bg_img = background_img.copy()
          if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
          if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
          b, g, r, a = cv2.split(img_to_overlay_t)   
          mask = cv2.medianBlur(a, 5)
          h, w, _ = img_to_overlay_t.shape
          roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
          img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
          img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)
          bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)
          bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
          return bg_img

        def angle_between(p1, p2):
          xDiff = p2[0] - p1[0]
          yDiff = p2[1] - p1[1]
          return degrees(atan2(yDiff, xDiff))
        
        faceCascade = cv2.CascadeClassifier(face_cascade_path)
        predictor = dlib.shape_predictor(predictor_path)

        image = cv2.imread(imagepath)
        image = cv2.resize(image, (800, 450))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray,(3,3), 0)
        faces = faceCascade.detectMultiScale(
            gauss,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(100,100),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        print("found {0} faces!".format(len(faces)) )
        
   
        for (x,y,w,h) in faces:
            #cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
         
            pre_detected_landmarks=predictor(image,dlib_rect) 
            detected_landmarks = pre_detected_landmarks.parts()
            landmarks = np.matrix([[p.x,p.y] for p in detected_landmarks])
           
            landmark = image.copy()
            for idx, point in enumerate(landmarks):
                    pos = (point[0,0], point[0,1] )
                    cv2.putText(landmark,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0,0,255) )
                    cv2.circle(landmark, pos, 3, color=(0,255,255))
        
        vec = np.empty([68, 2], dtype = int)   
        for b in range(68):
            vec[b][0] = pre_detected_landmarks.part(b).x 
            vec[b][1] = pre_detected_landmarks.part(b).y 

        ori_img=image.copy()
        result_img=image.copy()

        glasses_center = np.mean([vec[39], vec[42]], axis=0)  
        glasses_size = np.linalg.norm(vec[0] - vec[16])*0.95 #눈의 크기에 따라 안경 사이즈 조절

        angle =-angle_between(vec[39], vec[42])   

        M = cv2.getRotationMatrix2D((glasses.shape[1] / 2, glasses.shape[0] / 2), angle, 1)
        rotated_glasses = cv2.warpAffine(glasses, M, (glasses.shape[1],glasses.shape[0])) 
        try:
            result_img = overlay_transparent(result_img, rotated_glasses, glasses_center[0], glasses_center[1], overlay_size=(int(glasses_size), int(glasses.shape[0] * glasses_size / glasses.shape[1])))
        except:
            print('failed overlay image')

        cv2.imwrite('/Users/82103/Final_Project/Oval_fitted.png',result_img)

        self.qPixmapFileVar=QPixmap()
        self.qPixmapFileVar.load("/Users/82103/Final_Project/Oval_fitted.png")
        self.qPixmapFileVar = self.qPixmapFileVar.scaled(800,450)
        self.label.setPixmap(self.qPixmapFileVar)
        
#############Square############################################################################
        

    def Square_Function(self) :

        imagepath = "/Users/82103/FInal_Project/face_input.png"
        face_cascade_path = "/Users/82103/Final_Project/haarcascade_frontalface_default.xml"
        predictor_path = "/Users/82103/FInal_Project/shape_predictor_68_face_landmarks.dat"
        glasses = cv2.imread('/Users/82103/FInal_Project/glasses/round.png', cv2.IMREAD_UNCHANGED)
      
        def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
          bg_img = background_img.copy()
          if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
          if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
          b, g, r, a = cv2.split(img_to_overlay_t)   
          mask = cv2.medianBlur(a, 5)
          h, w, _ = img_to_overlay_t.shape
          roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
          img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
          img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)
          bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)
          bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
          return bg_img

        def angle_between(p1, p2):
          xDiff = p2[0] - p1[0]
          yDiff = p2[1] - p1[1]
          return degrees(atan2(yDiff, xDiff))
        
        faceCascade = cv2.CascadeClassifier(face_cascade_path)
        predictor = dlib.shape_predictor(predictor_path)

        image = cv2.imread(imagepath)
        image = cv2.resize(image, (800, 450))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray,(3,3), 0)
        faces = faceCascade.detectMultiScale(
            gauss,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(100,100),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        print("found {0} faces!".format(len(faces)) )
        
   
        for (x,y,w,h) in faces:
            #cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
         
            pre_detected_landmarks=predictor(image,dlib_rect) 
            detected_landmarks = pre_detected_landmarks.parts()
            landmarks = np.matrix([[p.x,p.y] for p in detected_landmarks])
           
            landmark = image.copy()
            for idx, point in enumerate(landmarks):
                    pos = (point[0,0], point[0,1] )
                    cv2.putText(landmark,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0,0,255) )
                    cv2.circle(landmark, pos, 3, color=(0,255,255))
        
        vec = np.empty([68, 2], dtype = int)   
        for b in range(68):
            vec[b][0] = pre_detected_landmarks.part(b).x 
            vec[b][1] = pre_detected_landmarks.part(b).y 

        ori_img=image.copy()
        result_img=image.copy()

        glasses_center = np.mean([vec[39], vec[42]], axis=0)  
        glasses_size = np.linalg.norm(vec[0] - vec[16])*0.95 #눈의 크기에 따라 안경 사이즈 조절

        angle =-angle_between(vec[39], vec[42])   

        M = cv2.getRotationMatrix2D((glasses.shape[1] / 2, glasses.shape[0] / 2), angle, 1)
        rotated_glasses = cv2.warpAffine(glasses, M, (glasses.shape[1],glasses.shape[0])) 
        try:
            result_img = overlay_transparent(result_img, rotated_glasses, glasses_center[0], glasses_center[1], overlay_size=(int(glasses_size), int(glasses.shape[0] * glasses_size / glasses.shape[1])))
        except:
            print('failed overlay image')

        cv2.imwrite('/Users/82103/Final_Project/Oval_fitted.png',result_img)

        self.qPixmapFileVar=QPixmap()
        self.qPixmapFileVar.load("/Users/82103/Final_Project/Oval_fitted.png")
        self.qPixmapFileVar = self.qPixmapFileVar.scaled(800,450)
        self.label.setPixmap(self.qPixmapFileVar)


#############Diamond############################################################################
        

    def Diamond_Function(self) :

        imagepath = "/Users/82103/FInal_Project/face_input.png"
        face_cascade_path = "/Users/82103/Final_Project/haarcascade_frontalface_default.xml"
        predictor_path = "/Users/82103/FInal_Project/shape_predictor_68_face_landmarks.dat"
        glasses = cv2.imread('/Users/82103/FInal_Project/glasses/eggplant.png', cv2.IMREAD_UNCHANGED)
      
        def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
          bg_img = background_img.copy()
          if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
          if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
          b, g, r, a = cv2.split(img_to_overlay_t)   
          mask = cv2.medianBlur(a, 5)
          h, w, _ = img_to_overlay_t.shape
          roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
          img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
          img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)
          bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)
          bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
          return bg_img

        def angle_between(p1, p2):
          xDiff = p2[0] - p1[0]
          yDiff = p2[1] - p1[1]
          return degrees(atan2(yDiff, xDiff))
        
        faceCascade = cv2.CascadeClassifier(face_cascade_path)
        predictor = dlib.shape_predictor(predictor_path)

        image = cv2.imread(imagepath)
        image = cv2.resize(image, (800, 450))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray,(3,3), 0)
        faces = faceCascade.detectMultiScale(
            gauss,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(100,100),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        print("found {0} faces!".format(len(faces)) )
        
   
        for (x,y,w,h) in faces:
            #cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
         
            pre_detected_landmarks=predictor(image,dlib_rect) 
            detected_landmarks = pre_detected_landmarks.parts()
            landmarks = np.matrix([[p.x,p.y] for p in detected_landmarks])
           
            landmark = image.copy()
            for idx, point in enumerate(landmarks):
                    pos = (point[0,0], point[0,1] )
                    cv2.putText(landmark,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0,0,255) )
                    cv2.circle(landmark, pos, 3, color=(0,255,255))
        
        vec = np.empty([68, 2], dtype = int)   
        for b in range(68):
            vec[b][0] = pre_detected_landmarks.part(b).x 
            vec[b][1] = pre_detected_landmarks.part(b).y 

        ori_img=image.copy()
        result_img=image.copy()

        glasses_center = np.mean([vec[39], vec[42]], axis=0)  
        glasses_size = np.linalg.norm(vec[0] - vec[16])*0.95 #눈의 크기에 따라 안경 사이즈 조절

        angle =-angle_between(vec[39], vec[42])   

        M = cv2.getRotationMatrix2D((glasses.shape[1] / 2, glasses.shape[0] / 2), angle, 1)
        rotated_glasses = cv2.warpAffine(glasses, M, (glasses.shape[1],glasses.shape[0])) 
        try:
            result_img = overlay_transparent(result_img, rotated_glasses, glasses_center[0], glasses_center[1], overlay_size=(int(glasses_size), int(glasses.shape[0] * glasses_size / glasses.shape[1])))
        except:
            print('failed overlay image')

        cv2.imwrite('/Users/82103/Final_Project/Oval_fitted.png',result_img)

        self.qPixmapFileVar=QPixmap()
        self.qPixmapFileVar.load("/Users/82103/Final_Project/Oval_fitted.png")
        self.qPixmapFileVar = self.qPixmapFileVar.scaled(800,450)
        self.label.setPixmap(self.qPixmapFileVar)

#############Triangle############################################################################
        

    def Triangle_Function(self) :

        imagepath = "/Users/82103/FInal_Project/face_input.png"
        face_cascade_path = "/Users/82103/Final_Project/haarcascade_frontalface_default.xml"
        predictor_path = "/Users/82103/FInal_Project/shape_predictor_68_face_landmarks.dat"
        glasses = cv2.imread('/Users/82103/FInal_Project/glasses/wellington10.png', cv2.IMREAD_UNCHANGED)
      
        def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
          bg_img = background_img.copy()
          if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
          if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
          b, g, r, a = cv2.split(img_to_overlay_t)   
          mask = cv2.medianBlur(a, 5)
          h, w, _ = img_to_overlay_t.shape
          roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
          img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
          img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)
          bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)
          bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
          return bg_img

        def angle_between(p1, p2):
          xDiff = p2[0] - p1[0]
          yDiff = p2[1] - p1[1]
          return degrees(atan2(yDiff, xDiff))
        
        faceCascade = cv2.CascadeClassifier(face_cascade_path)
        predictor = dlib.shape_predictor(predictor_path)

        image = cv2.imread(imagepath)
        image = cv2.resize(image, (800, 450))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray,(3,3), 0)
        faces = faceCascade.detectMultiScale(
            gauss,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(100,100),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        print("found {0} faces!".format(len(faces)) )
        
   
        for (x,y,w,h) in faces:
            #cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
         
            pre_detected_landmarks=predictor(image,dlib_rect) 
            detected_landmarks = pre_detected_landmarks.parts()
            landmarks = np.matrix([[p.x,p.y] for p in detected_landmarks])
           
            landmark = image.copy()
            for idx, point in enumerate(landmarks):
                    pos = (point[0,0], point[0,1] )
                    cv2.putText(landmark,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0,0,255) )
                    cv2.circle(landmark, pos, 3, color=(0,255,255))
        
        vec = np.empty([68, 2], dtype = int)   
        for b in range(68):
            vec[b][0] = pre_detected_landmarks.part(b).x 
            vec[b][1] = pre_detected_landmarks.part(b).y 

        ori_img=image.copy()
        result_img=image.copy()

        glasses_center = np.mean([vec[39], vec[42]], axis=0)  
        glasses_size = np.linalg.norm(vec[0] - vec[16])*0.95 #눈의 크기에 따라 안경 사이즈 조절

        angle =-angle_between(vec[39], vec[42]) 

        M = cv2.getRotationMatrix2D((glasses.shape[1] / 2, glasses.shape[0] / 2), angle, 1)
        rotated_glasses = cv2.warpAffine(glasses, M, (glasses.shape[1],glasses.shape[0])) 
        try:
            result_img = overlay_transparent(result_img, rotated_glasses, glasses_center[0], glasses_center[1], overlay_size=(int(glasses_size), int(glasses.shape[0] * glasses_size / glasses.shape[1])))
        except:
            print('failed overlay image')

        cv2.imwrite('/Users/82103/Final_Project/Oval_fitted.png',result_img)

        self.qPixmapFileVar=QPixmap()
        self.qPixmapFileVar.load("/Users/82103/Final_Project/Oval_fitted.png")
        self.qPixmapFileVar = self.qPixmapFileVar.scaled(800,450)
        self.label.setPixmap(self.qPixmapFileVar)

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass() 
    myWindow.show()
    
    app.exec_()
