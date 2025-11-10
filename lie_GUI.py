import time
import cv2
import numpy as np
import sys
import os
import argparse
from random import sample
#Landmark load model
import yaml
#PyQt5
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot, QTimer, QThread, QSize, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QBrush
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QGraphicsOpacityEffect
from PyQt5 import QtGui
import Qt_design as ui
from threading import Thread
#torch
import torch
import torchvision
#image transform
import skimage
import imageio
from skimage import img_as_ubyte
from model.Facedetection.config import device
#face detection
from model.Facedetection.utils import align_face, get_face_all_attributes, draw_bboxes
from model.Facedetection.RetinaFace.RetinaFaceDetection import retina_face
#Featrue extraction
import model.Emotion.lie_emotion_process as emotion
import model.action_v4_L12_BCE_MLSM.lie_action_process as action
from model.action_v4_L12_BCE_MLSM.config import Config
#Landmark
# from model.Landmark.TDDFA import TDDFA
# from model.Landmark.utils.render import render
# from model.Landmark.utils.functions import cv_draw_landmark, get_suffix
# save model
from joblib import dump, load

parser = argparse.ArgumentParser()
#Retina
parser.add_argument('--len_cut', default=30, type=int, help= '# of frames you want to pred')
parser.add_argument('-m', '--trained_model', default='./model/Facedetection/RetinaFace/weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=3000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=3, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--gpu_num', default= "0", type=str, help='GPU number')
#Landmark
parser.add_argument('-c', '--config', type=str, default='./model/Landmark/configs/mb1_120x120.yml')
parser.add_argument('--mode', default='gpu', type=str, help='gpu or cpu mode')
parser.add_argument('-o', '--opt', type=str, default='2d', choices=['2d', '3d'])
# Emotion
parser.add_argument('--at_type', '--attention', default=1, type=int, metavar='N',help= '0 is self-attention; 1 is self + relation-attention')
parser.add_argument('--preTrain_path', '-pret', default='./model/Emotion/model112/self_relation-attention_AFEW_better_46.0733_41.2759_12.tar', type=str, help='pre-training model path')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

#load model
Retina = retina_face(crop_size = 224, args = args) # Face detection
Emotion_class = emotion.Emotion_FAN(args = args)
Action_class = action.Action_Resnet(args= Config())
SVM_model = load('./model/SVM_model/se_res50+EU/split_svc_acc0.720_AUC0.828.joblib')
print('model is loaded')
class Landmark:
    def __init__(self,im,bbox,cfg,TDDFA,color):
        self.cfg = cfg
        self.tddfa = TDDFA
        self.boxes = bbox
        self.image = im
        self.color = color
        
    def main(self,index):
        dense_flag = args.opt in ('3d',)
        pre_ver = None
        self.boxes = [self.boxes[index]]
        param_lst, roi_box_lst = self.tddfa(self.image, self.boxes)
        ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        # refine
        param_lst, roi_box_lst = self.tddfa(self.image, [ver], crop_policy='landmark')
        ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        pre_ver = ver  # for tracking

        if args.opt == '2d':
            res = cv_draw_landmark(self.image, ver,color=self.color)
        elif args.opt == '3d':
            res = render(self.image, [ver])
        else:
            raise Exception(f'Unknown opt {args.opt}')
        
        lnd = ver.T
        # D1_i = np.sqrt(np.square(lnd[61][0]-lnd[67][0]) + np.square(lnd[61][1]-lnd[67][1]))
        # D1_o = np.sqrt(np.square(lnd[50][0]-lnd[58][0]) + np.square(lnd[50][1]-lnd[58][1]))
        # D2_i = np.sqrt(np.square(lnd[62][0]-lnd[66][0]) + np.square(lnd[62][1]-lnd[66][1]))
        # D2_o = np.sqrt(np.square(lnd[51][0]-lnd[57][0]) + np.square(lnd[51][1]-lnd[57][1]))
        # D3_i = np.sqrt(np.square(lnd[63][0]-lnd[65][0]) + np.square(lnd[63][1]-lnd[65][1]))
        # D3_o = np.sqrt(np.square(lnd[52][0]-lnd[56][0]) + np.square(lnd[52][1]-lnd[56][1]))
        res = res[int(roi_box_lst[0][1]):int(roi_box_lst[0][3]), int(roi_box_lst[0][0]):int(roi_box_lst[0][2])]
        # pm_ratio_1 = D1_i / D1_o
        # pm_ratio_2 = D2_i / D2_o
        # pm_ratio_3 = D3_i / D3_o
        # print('pm1:',pm_ratio_1)
        # print('pm2:',pm_ratio_2)
        # print('pm3:',pm_ratio_3)
        if res.shape[0] != 0 and res.shape[1] != 0:
            img_res = cv2.resize(res,(224,224))
        else:
            img_res = np.array([None])
        return img_res

#AU_pred thread
class AU_pred(QThread):
    trigger = pyqtSignal(list,list)
    def  __init__ (self,image):
        super(AU_pred ,self). __init__ ()
        self.face = image
    def run(self):
        logps, emb = Action_class._pred(self.face,Config)
        self.trigger.emit(emb.tolist(),logps.tolist())

class show(QThread):
    # now emits logps (list), pred_score (list), results (int scalar), prob (float 0..1)
    trigger = pyqtSignal(list, list, int, float)

    def __init__(self, frame_list, frame_AU, log):
        super(show, self).__init__()
        # frame_list: list of face crops (for emotion model)
        # frame_AU: numpy array (AU features averaged) or list-like
        # log: AU logits list (or array)
        self.frame_embed_list = frame_list
        # Ensure frame_AU is a numpy array (1D)
        self.frame_emb_AU = np.asarray(frame_AU).reshape(1, -1) if frame_AU is not None else None
        self.log = log

    def pred(self):
        """
        - produce AU_list (binary AU presence)
        - produce pred_score (emotion prediction vector from Emotion model)
        - compute feature for SVM by concatenating AU features and relation embedding
        - return (AU_list, pred_score, results_scalar, prob_float)
        """
        # Action calculation -> binarize AU logits (self.log)
        AU_list = None
        try:
            # self.log may be a numpy array or list-like of shape (1,12)
            log_arr = np.asarray(self.log)
            if log_arr.ndim == 2 and log_arr.shape[0] == 1:
                AU_vals = log_arr[0]
            else:
                AU_vals = log_arr
            AU_list = [1 if float(x) >= 0.01 else 0 for x in AU_vals]
        except Exception:
            # fallback: zeros
            AU_list = [0] * 12

        # Emotion prediction - call Emotion_class.validate
        pred_score, self_embedding, relation_embedding = Emotion_class.validate(self.frame_embed_list)
        # relation_embedding expected as a torch tensor on CPU
        if hasattr(relation_embedding, 'cpu'):
            rel_emb_np = relation_embedding.cpu().numpy().reshape(1, -1)
        else:
            rel_emb_np = np.asarray(relation_embedding).reshape(1, -1)

        # frame_emb_AU should already be shaped (1, k)
        if self.frame_emb_AU is None:
            # fallback: zeros with correct dim if unknown
            self.frame_emb_AU = np.zeros((1, rel_emb_np.shape[1]//2))  # fallback heuristic

        # Build feature: concatenate horizontally -> shape (1, total_features)
        try:
            feature = np.concatenate((self.frame_emb_AU.astype(np.float32), rel_emb_np.astype(np.float32)), axis=1)
        except Exception as e:
            print(f"[DEBUG] Feature concatenation failed: {e}")
            # fallback: try flatten and stack
            feature = np.hstack([np.asarray(self.frame_emb_AU).reshape(1, -1), np.asarray(rel_emb_np).reshape(1, -1)])

        # Predict with SVM (robustly)
        results_scalar = 0
        prob = 0.0
        try:
            # ensure feature is float64 for sklearn compatibility
            feature_sklearn = feature.astype(np.float64)
            if hasattr(SVM_model, "predict_proba"):
                probs = SVM_model.predict_proba(feature_sklearn)  # shape (1, n_classes)
                # We assume class order: SVM_model.classes_ -> [0,1] or similar
                # Take probability of class '1' (deception) if possible
                if probs.shape[1] == 1:
                    prob = float(probs[0, 0])
                elif probs.shape[1] >= 2:
                    # try to find index of class '1'
                    try:
                        idx1 = list(SVM_model.classes_).index(1)
                        prob = float(probs[0, idx1])
                    except Exception:
                        # default: take the max probability for predicted class
                        prob = float(np.max(probs[0]))
                results_arr = SVM_model.predict(feature_sklearn)
                results_scalar = int(np.asarray(results_arr).reshape(-1)[0])
            else:
                # no predict_proba: try decision_function -> sigmoid
                if hasattr(SVM_model, "decision_function"):
                    df = SVM_model.decision_function(feature_sklearn)
                    df0 = float(np.asarray(df).reshape(-1)[0])
                    prob = 1.0 / (1.0 + np.exp(-df0))
                else:
                    # last resort: use predict and set prob = 1.0 for predicted class
                    pred_arr = SVM_model.predict(feature_sklearn)
                    results_scalar = int(np.asarray(pred_arr).reshape(-1)[0])
                    prob = 1.0
                # if results_scalar still zero, get predicted class now:
                if not results_scalar:
                    try:
                        results_scalar = int(np.asarray(SVM_model.predict(feature_sklearn)).reshape(-1)[0])
                    except Exception:
                        results_scalar = 0
        except Exception as e:
            print(f"[DEBUG] SVM prediction error: {e}")
            # fallback defaults
            try:
                # try simple direct predict
                pred_arr = SVM_model.predict(feature)
                results_scalar = int(np.asarray(pred_arr).reshape(-1)[0])
            except Exception:
                results_scalar = 0
            prob = 0.0

        # Debug prints (helpful while testing)
        print(f"[DEBUG] SVM classes: {getattr(SVM_model, 'classes_', 'N/A')}")
        print(f"[DEBUG] feature.shape: {feature.shape}, results: {results_scalar}, prob: {prob:.4f}")

        return AU_list, pred_score, int(results_scalar), float(prob)

    def run(self):
        logps, pred_score, results, prob = self.pred()
        # emit logps (list), pred_score (list), results (int), prob (float)
        self.trigger.emit(logps, pred_score.tolist() if hasattr(pred_score, 'tolist') else pred_score, int(results), float(prob))        

class lie_GUI(QDialog, ui.Ui_Dialog):
    def __init__(self, args):
        super(lie_GUI, self).__init__()
        print('Start deception detection')
        import qdarkstyle
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.mouth_count = 0
        self.frame_embed_list = [] # 儲存人臉
        self.frame_emb_AU = []
        self.log = []
        self.userface =[]
        self.color = (0, 255, 0)
        self.index = 0
        self.len_bbox = 1
        self.time = None 
        #Qt_design
        self.setupUi(self)
        self.Startlabel.setText('Press the button to upload a video or activate camera')
        self.Problem.setPlaceholderText('Enter the question')
        self.Record.setPlaceholderText('Enter the description')
        #hidden button
        self.Reset.setVisible(False)
        self.Finish.setVisible(False)
        self.truth_lie.setVisible(False)
        self.prob_label.setVisible(False)
        self.Start.setVisible(False)
        self.RecordStop.setVisible(False)
        self.filename.setVisible(False)
        self.videoprogress.setVisible(False)
        self.User0.setVisible(False)
        self.User1.setVisible(False)
        self.User2.setVisible(False)
        self.Record_area.setVisible(False)
        self.Problem.setVisible(False)
        self.Record.setVisible(False)
        # self.Export.setVisible(False)
        self.camera_start.setVisible(False)
        self.Clear.setVisible(False)
        self.camera_finish.setVisible(False)
        #set style
        self.videoprogress.setStyleSheet("QProgressBar::chunk ""{""background-color: white;""}") ##4183c5
        #button click
        self.loadcamera.clicked.connect(self.start_webcam)
        self.loadvideo.clicked.connect(self.get_image_file)
        self.Reset.clicked.connect(self.Reset_but)
        self.Finish.clicked.connect(self.Reset_but)
        self.camera_finish.clicked.connect(self.Reset_but)
        self.Start.clicked.connect(self.time_start)
        self.RecordStop.clicked.connect(self.record_stop)
        self.camera_start.clicked.connect(self.Enter_problem)
        self.Clear.clicked.connect(self.cleartext)
        self.User0.clicked.connect(self.User_0)
        self.User1.clicked.connect(self.User_1)
        self.User2.clicked.connect(self.User_2)
        #button icon
        self.loadvideo.setIcon(QIcon('./icon/youtube.png')) # set button icon
        self.loadvideo.setIconSize(QSize(50,50)) # set icon size
        self.loadcamera.setIcon(QIcon('./icon/camera.png')) # set button icon
        self.loadcamera.setIconSize(QSize(50,50)) # set icon size
        self.Reset.setIcon(QIcon('./icon/reset.png')) # set button icon
        self.Reset.setIconSize(QSize(60,60)) # set icon size
        self.RecordStop.setIcon(QIcon('./icon/stop.png')) # set button icon
        self.RecordStop.setIconSize(QSize(30,30)) # set icon size
        #Landmark
        # self.cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
        # self.tddfa = TDDFA(gpu_mode='gpu', **self.cfg)
        #攝像頭
        self.cap = None
        self.countframe = 0
        #timer
        self.timer = QTimer(self, interval=0)
        self.timer.timeout.connect(self.update_frame)

    def cleartext(self):
        self.Problem.clear()
        self.Record.clear()

    def Enter_problem(self):
        the_input = self.Problem.toPlainText() #文字框的字
        
        with open('Result.txt', 'a',newline='') as f:
            f.write("Problem :")
            f.write(the_input)
            
        self.time_start()

    def User_0(self):
        self.User0.setVisible(False)
        self.User1.setVisible(False)
        self.User2.setVisible(False)
        self.index = 0
        self.userface = self.face_list[self.index]
        self.userface = np.array(self.userface)
        self.Startlabel.setVisible(False)
        self.timer.start()
        self.RecordStop.setVisible(True)
        
    def User_1(self):
        self.User0.setVisible(False)
        self.User1.setVisible(False)
        self.User2.setVisible(False)
        self.index = 1
        self.userface = self.face_list[self.index]
        self.userface = np.array(self.userface)
        self.Startlabel.setVisible(False)
        self.timer.start()
        self.RecordStop.setVisible(True)

    def User_2(self):
        self.User0.setVisible(False)
        self.User1.setVisible(False)
        self.User2.setVisible(False)
        self.index = 2
        self.userface = self.face_list[self.index]
        self.userface = np.array(self.userface)
        self.Startlabel.setVisible(False)
        self.timer.start()
        self.RecordStop.setVisible(True)

    def time_start(self):
        if self.cap is not None:
            if self.mode == 'camera':
                self.Start.setVisible(False)
                self.RecordStop.setVisible(True)
            else:
                self.Start.setVisible(False)
                self.RecordStop.setVisible(True)
                self.videoprogress.setVisible(True)
            #把所有歸零
            self.User0.setVisible(False)
            self.User1.setVisible(False)
            self.User2.setVisible(False)
            self.camera_finish.setVisible(False)
            self.camera_start.setVisible(False)
            self.Clear.setVisible(False)
            self.prob_label.setVisible(False)
            self.Reset.setVisible(False)
            self.timer.start()
            self.truth_lie.setVisible(False)
            self.A01.setStyleSheet('''color:#c3c3c3''')
            self.A02.setStyleSheet('''color:#c3c3c3''')
            self.A04.setStyleSheet('''color:#c3c3c3''')
            self.A05.setStyleSheet('''color:#c3c3c3''')
            self.A06.setStyleSheet('''color:#c3c3c3''')
            self.A09.setStyleSheet('''color:#c3c3c3''')
            self.A12.setStyleSheet('''color:#c3c3c3''')
            self.A15.setStyleSheet('''color:#c3c3c3''')
            self.A17.setStyleSheet('''color:#c3c3c3''')
            self.A20.setStyleSheet('''color:#c3c3c3''')
            self.A25.setStyleSheet('''color:#c3c3c3''')
            self.A26.setStyleSheet('''color:#c3c3c3''')
            self.Happly_label.setStyleSheet('''color:#c3c3c3''')
            self.Angry_label.setStyleSheet('''color:#c3c3c3''')
            self.DIsgust_label.setStyleSheet('''color:#c3c3c3''')
            self.Fear_label.setStyleSheet('''color:#c3c3c3''')
            self.Sad_label.setStyleSheet('''color:#c3c3c3''')
            self.Neutral_label.setStyleSheet('''color:#c3c3c3''')
            self.Surprise_label.setStyleSheet('''color:#c3c3c3''')


    def record_stop(self):
        self.timer.stop()
        if self.mode =='video':
            self.RecordStop.setVisible(False)
            self.Start.setVisible(True)
            self.Finish.setVisible(True)
        else:
            self.frame_emb_AU = np.array(self.frame_emb_AU)
            self.frame_emb_AU = np.mean(self.frame_emb_AU, axis = 0)
            self.log = np.array(self.log)
            self.log = np.mean(self.log, axis = 0)
            self.show_thread = show(self.frame_embed_list, self.frame_emb_AU,self.log)
            self.show_thread.start()
            self.show_thread.trigger.connect(self.display_feature)
            self.frame_embed_list = []
            self.frame_emb_AU = []
            self.log = []
            self.camera_finish.setVisible(True)
            _translate = QtCore.QCoreApplication.translate
            self.camera_start.setText(_translate("Dialog", "Continue"))
            self.camera_start.setVisible(True)
            self.Clear.setVisible(True)
            self.RecordStop.setVisible(False)
            self.camera_finish.setVisible(True)

        self.Reset.setVisible(False)
        
        if self.lie_count > 0:
            lie_prob = round((self.lie_prob_count / self.lie_count) * 100)
        else:
            lie_prob = 0
        self.RecordStop.setVisible(False)
        self.prob_label.setVisible(True)
        self.prob_label.setText(f'The probability of deception: {lie_prob}%')


    def start_webcam(self):
        self.lie_count = 0
        self.lie_prob_count = 0
        self.loadvideo.setVisible(False)
        self.loadcamera.setVisible(False)
        # self.Start.setVisible(True)
        self.Reset.setVisible(True)
        if self.cap is None:
            self.Startlabel.setVisible(False)
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.mode = 'camera'
            self.Problem.setVisible(True)
            self.Record.setVisible(True)
            self.Record_area.setVisible(True)

            self.Clear.setVisible(True)
            self.camera_start.setVisible(True)
        with open('Result.txt', 'w',newline='') as f:
            f.write("\t\t\t\t\t\tReport\n")
            

    def get_image_file(self):
        self.lie_count = 0
        self.lie_prob_count = 0
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Video File', r"<Default dir>", "Video files (*.mp4 *.avi)")
        if self.cap is None and file_name != '':
            self.Startlabel.setVisible(False)
            self.loadvideo.setVisible(False)
            self.loadcamera.setVisible(False)
            self.Start.setVisible(True)
            self.filename.setVisible(True)
            self.Finish.setVisible(False)
            self.Reset.setVisible(True)
            self.filename.setText('        Current file:\n{:^29}' .format(file_name.split('/')[-1]))
            self.cap = cv2.VideoCapture(file_name)
            self.frame_total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.mode = 'video'
    
    def face_recognition(self,bbox,img):
        self.Original.setPixmap(QPixmap(""))
        # self.Facealignment.setPixmap(QPixmap(""))
        # self.Landmark.setPixmap(QPixmap(""))

        qformat = QImage.Format_Indexed8
        if len(img.shape)==3 :
            if img.shape[2]==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img_raw = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        img_raw = img_raw.rgbSwapped()
        self.Original.setPixmap(QPixmap.fromImage(img_raw))

        self.Startlabel.setVisible(True)
        self.RecordStop.setVisible(False)

        if len(bbox) == 2:
            op0 = QGraphicsOpacityEffect()
            op0.setOpacity(0)
            op1 = QGraphicsOpacityEffect()
            op1.setOpacity(0)
            self.User0.setGraphicsEffect(op0)
            self.User1.setGraphicsEffect(op1)
            self.User0.setGeometry(QtCore.QRect(bbox[0][0], bbox[0][1]+25, 251, 91))
            self.User1.setGeometry(QtCore.QRect(bbox[1][0], bbox[1][1]+25, 251, 91))
            self.User0.setVisible(True)
            self.User1.setVisible(True)

        elif len(bbox) == 3:
            op0 = QGraphicsOpacityEffect()
            op0.setOpacity(0)
            op1 = QGraphicsOpacityEffect()
            op1.setOpacity(0)
            op2 = QGraphicsOpacityEffect()
            op2.setOpacity(0)
            self.User0.setGraphicsEffect(op0)
            self.User1.setGraphicsEffect(op1)
            self.User0.setGeometry(QtCore.QRect(bbox[0][0], bbox[0][1]+25, 251, 91))
            self.User1.setGeometry(QtCore.QRect(bbox[1][0], bbox[1][1]+25, 251, 91))
            self.User2.setGeometry(QtCore.QRect(bbox[2][0], bbox[2][1]+25, 251, 91))
            self.User0.setVisible(True)
            self.User1.setVisible(True)
            self.User2.setVisible(True)
        self.Startlabel.setText('Choose the user you want to detect!')
        
        

    def update_frame(self):
        ret, im = self.cap.read()
        
        # CRITICAL FIX: Validate frame was captured successfully
        if not ret or im is None:
            print("Warning: Failed to read frame from camera/video")
            if self.mode == 'video':
                # Video ended, stop timer and show results
                self.timer.stop()
                self.prob_label.setVisible(True)
                self.RecordStop.setVisible(False)
                self.Reset.setVisible(True)
                if self.lie_count != 0:
                    lie_prob = round((self.lie_prob_count / self.lie_count) * 100)
                    self.prob_label.setText('The probability of deception: {:.0f}% '.format(lie_prob))
            return
        
        # Validate frame dimensions
        if im.size == 0 or im.shape[0] == 0 or im.shape[1] == 0:
            print(f"Warning: Invalid frame dimensions - shape: {im.shape}")
            return
        
        # Now safe to resize
        try:
            im = cv2.resize(im, (640, 480), interpolation=cv2.INTER_AREA)
        except cv2.error as e:
            print(f"Error resizing frame: {e}")
            return
            
        self.im = im
        show_img = im.copy()
        self.countframe += 1
        
        #影片讀條
        if self.mode == 'video':
            self.videoprogress.setValue(int(round(self.countframe / self.frame_total * 100)))

        image = skimage.img_as_float(im).astype(np.float32)
        frame = img_as_ubyte(image)
        result = Retina.detect_face(frame)
        if len(result) == 4:
            self.img_raw, output_raw, output_points, bbox = result
            self.face_list = [bbox] if bbox is not None else []
        else:
            self.img_raw, output_raw, output_points, bbox, self.face_list = result
        #若只有一個臉，正常顯示
        if len(bbox) == 1:
            self.index = 0
            self.userface = self.face_list[self.index]
            self.len_bbox = 1
        elif len(bbox) >= 2:
            if len(self.userface):
                dist_list = []
                self.face_list = np.array(self.face_list)
                for i in range(len(bbox)):
                    if i < self.face_list.shape[0]:
                        dist = np.sqrt(np.sum(np.square(np.subtract(self.userface[:], self.face_list[i, :]))))
                    else:
                        continue
                    dist_list.append(dist)
                dist_list = np.array(dist_list)
                self.index = np.argmin(dist_list)

        if(len(output_points)):
            #face_align
            out_raw = align_face(output_raw, output_points[self.index], crop_size_h = 112, crop_size_w = 112)
            out_raw = cv2.resize(out_raw,(224, 224))
            #Landmark
            # _landmark = Landmark(im,bbox,self.cfg,self.tddfa,self.color)
            # landmark_img = _landmark.main(self.index)

            cv2.rectangle(show_img, (bbox[self.index][0], bbox[self.index][1]), (bbox[self.index][2], bbox[self.index][3]), (0, 0, 255), 2)
            self.face_align = out_raw
            self.bbox = bbox
            self.displayImage(show_img,bbox,True)
            self.frame_embed_list.append(out_raw) #儲存人臉
            #計算AU
            self.lnd_AU = AU_pred(out_raw)
            self.lnd_AU.start()
            self.lnd_AU.trigger.connect(self.AU_store)
        #沒有臉的時候
        else:
            self.frame_embed_list = []
            self.frame_emb_AU = []
            self.log = []
            self.A01.setStyleSheet('''color:#e8e8e8''')  # #e8e8e8
            self.A02.setStyleSheet('''color:#e8e8e8''')
            self.A04.setStyleSheet('''color:#e8e8e8''')
            self.A05.setStyleSheet('''color:#e8e8e8''')
            self.A06.setStyleSheet('''color:#e8e8e8''')
            self.A09.setStyleSheet('''color:#e8e8e8''')
            self.A12.setStyleSheet('''color:#e8e8e8''')
            self.A15.setStyleSheet('''color:#e8e8e8''')
            self.A17.setStyleSheet('''color:#e8e8e8''')
            self.A20.setStyleSheet('''color:#e8e8e8''')
            self.A25.setStyleSheet('''color:#e8e8e8''')
            self.A26.setStyleSheet('''color:#e8e8e8''')
            self.Happly_label.setStyleSheet('''color:#e8e8e8''')
            self.Angry_label.setStyleSheet('''color:#e8e8e8''')
            self.DIsgust_label.setStyleSheet('''color:#e8e8e8''')
            self.Fear_label.setStyleSheet('''color:#e8e8e8''')
            self.Sad_label.setStyleSheet('''color:#e8e8e8''')
            self.Neutral_label.setStyleSheet('''color:#e8e8e8''')
            self.Surprise_label.setStyleSheet('''color:#e8e8e8''')
            self.truth_lie.setVisible(False)
            self.displayImage(im, face_num = None)
            
    def AU_store(self,AU_emb,log):
        AU_emb = torch.FloatTensor(AU_emb)
        log = torch.FloatTensor(log)
        # print(log)
        self.frame_emb_AU.append(AU_emb.cpu().numpy())
        self.log.append(log.cpu().numpy())

    def displayImage(self, img,bbox=None,face_num = None ):
        #定義參數
        qformat = QImage.Format_Indexed8
        if len(img.shape)==3 :
            if img.shape[2]==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        #顯示影像
        if face_num:
            img_raw = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
            img_raw = img_raw.rgbSwapped()
            # if lnd_img.any() != None:
            #     landmark_image = QImage(lnd_img, lnd_img.shape[1], lnd_img.shape[0], lnd_img.strides[0], qformat)
            #     landmark_image = landmark_image.rgbSwapped()
            #     self.Landmark.setPixmap(QPixmap.fromImage(landmark_image))

            # align_img = QImage(face_align, face_align.shape[1], face_align.shape[0], face_align.strides[0], qformat)
            # align_img = align_img.rgbSwapped()
            
            self.Original.setPixmap(QPixmap.fromImage(img_raw))
            # self.Facealignment.setPixmap(QPixmap.fromImage(align_img))
            
            #若大於兩個人，則選擇要哪個人
            if len(bbox) >=2 and self.len_bbox != len(bbox):
                self.frame_embed_list = []
                self.frame_emb_AU = []
                self.log = []
                self.len_bbox = len(bbox)
                self.timer.stop()
                self.face_recognition(bbox,self.img_raw)
            #若是影片，則len_cut禎計算一次結果
            if self.mode =='video':
                if len(self.frame_embed_list) == args.len_cut:
                    self.frame_emb_AU = np.array(self.frame_emb_AU)
                    self.frame_emb_AU = np.mean(self.frame_emb_AU, axis = 0)
                    self.log = np.array(self.log)
                    self.log = np.mean(self.log, axis = 0)
                    self.show_thread = show(self.frame_embed_list, self.frame_emb_AU,self.log)
                    self.show_thread.start()
                    self.show_thread.trigger.connect(self.display_feature)
                    self.frame_embed_list = []
                    self.frame_emb_AU = []
                    self.log = []

                
        else:
            img_raw = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
            img_raw = img_raw.rgbSwapped()
            self.Original.setPixmap(QPixmap.fromImage(img_raw))
            # self.Facealignment.setPixmap(QPixmap(""))
            # self.Landmark.setPixmap(QPixmap(""))
            if self.mode =='camera' :
                    self.countframe =0

        if self.mode =='video' :
            if self.countframe == self.frame_total:
                self.prob_label.setVisible(True)
                self.RecordStop.setVisible(False)
                self.Reset.setVisible(True)
                if self.lie_count != 0:
                    lie_prob = round((self.lie_prob_count / self.lie_count) * 100)
                    self.prob_label.setText('The probability of deception: {:.0f}% '.format(lie_prob))
                self.timer.stop()
    def display_feature(self, logps, pred_score, results, prob):
        """
        logps: list of AU binary flags (12)
        pred_score: list or tensor - emotion probs
        results: int scalar (0 or 1)
        prob: float in [0,1] representing SVM probability for class 1 (deception)
        """
        # Ensure results is a pure integer scalar
        try:
            if isinstance(results, (list, tuple, np.ndarray)):
                results = int(np.asarray(results).reshape(-1)[0])
            elif torch.is_tensor(results):
                results = int(results.item())
            else:
                results = int(results)
        except Exception:
            results = 0

        # For pred_score make a tensor so argmax works the same
        try:
            pred_score_t = torch.Tensor(pred_score) if not torch.is_tensor(pred_score) else pred_score
        except Exception:
            pred_score_t = torch.zeros(7)

        # Reset styles
        for au in ['A01','A02','A04','A05','A06','A09','A12','A15','A17','A20','A25','A26']:
            getattr(self, au).setStyleSheet('''color:#c3c3c3''')
        for lbl in ['Happly_label','Angry_label','DIsgust_label','Fear_label','Sad_label','Neutral_label','Surprise_label']:
            getattr(self, lbl).setStyleSheet('''color:#c3c3c3''')

        self.truth_lie.setText('')

        # Update UI based on results
        if results == 1:
            self.color = (0, 0, 255)  # red
            self.truth_lie.setText('Deception!')
            self.truth_lie.setStyleSheet('''QPushButton{background:#fff;border-radius:5px;color: red;}''')
            self.truth_lie.setVisible(True)
            # mark emotion and AUs in red
            emo_idx = int(pred_score_t.cpu().numpy().argmax()) if hasattr(pred_score_t, 'cpu') else int(pred_score_t.numpy().argmax())
            # safe mapping:
            emo_map = {0: 'Happly_label', 1: 'Angry_label', 2: 'DIsgust_label', 3: 'Fear_label', 4: 'Sad_label', 5: 'Neutral_label', 6: 'Surprise_label'}
            if emo_idx in emo_map:
                getattr(self, emo_map[emo_idx]).setStyleSheet('''color:red''')
            # action units
            for i,au_flag in enumerate(logps):
                if au_flag == 1:
                    getattr(self, ['A01','A02','A04','A05','A06','A09','A12','A15','A17','A20','A25','A26'][i]).setStyleSheet('''color:red''')
            # accumulate lie prob
            self.lie_prob_count += float(prob) * 100.0
        else:
            self.color = (0, 255, 0)  # green
            self.truth_lie.setText('Truth!')
            self.truth_lie.setStyleSheet('''QPushButton{background:#fff;border-radius:5px;color: green;}''')
            self.truth_lie.setVisible(True)
            emo_idx = int(pred_score_t.cpu().numpy().argmax()) if hasattr(pred_score_t, 'cpu') else int(pred_score_t.numpy().argmax())
            emo_map = {0: 'Happly_label', 1: 'Angry_label', 2: 'DIsgust_label', 3: 'Fear_label', 4: 'Sad_label', 5: 'Neutral_label', 6: 'Surprise_label'}
            if emo_idx in emo_map:
                getattr(self, emo_map[emo_idx]).setStyleSheet('''color:green''')
            for i,au_flag in enumerate(logps):
                if au_flag == 1:
                    getattr(self, ['A01','A02','A04','A05','A06','A09','A12','A15','A17','A20','A25','A26'][i]).setStyleSheet('''color:green''')

        # Bookkeeping
        # We increment lie_count by 1 for every call (this mirrors prior behavior)
        self.lie_count = getattr(self, 'lie_count', 0) + 1

        # Compute running average (lie_prob_count stores % as float sum of percentages)
        if self.lie_count > 0:
            avg_prob = (self.lie_prob_count / self.lie_count)
        else:
            avg_prob = 0.0

        # Update the GUI probability label (show percent with 1 decimal)
        self.prob_label.setVisible(True)
        try:
            self.prob_label.setText(f"The probability of deception: {avg_prob:.1f}%")
        except Exception:
            self.prob_label.setText(f"The probability of deception: {avg_prob:.1f}%")

        # Reset per-window buffers (same as original logic)
        self.frame_embed_list = []
        self.frame_emb_AU = []
        # keep lie_count (already incremented)
        # If in camera mode, reset countframe
        if getattr(self, 'mode', None) == 'camera':
            self.countframe = 0

        # write to Result.txt as before (keeps your original logging)
        try:
            with open('Result.txt', 'a', newline='') as f:
                f.write("\nEmotion unit:")
                emo_idx = int(pred_score_t.cpu().numpy().argmax())
                emo_label = ['Happy','Angry','Disgust','Fear','Sad','Neutral','Surprise'][emo_idx]
                f.write(emo_label)
                f.write('\nAction unit:')
                AU_names = ['Inner brow raiser','Outer brow raiser','Brow lower','Upper Lid Raiser','Cheek raiser','Nose wrinkle','Lip corner puller','Lip corner depressor','Chin raiser','Lip Stretcher','Lips part','Jaw drop']
                for i,au_flag in enumerate(logps):
                    if au_flag == 1:
                        f.write(AU_names[i] + '\t')
                f.write('\nLie detection:')
                f.write('Deception!' if results == 1 else 'Truth!')
                the_output = self.Record.toPlainText() if hasattr(self, 'Record') else ''
                f.write('\nDescription:')
                f.write(the_output)
                f.write('\n\n')
        except Exception:
            pass


    def Reset_but(self):
        self.Reset.setVisible(False)
        self.camera_finish.setVisible(False)
        self.Finish.setVisible(False)
        self.truth_lie.setVisible(False)
        self.videoprogress.setVisible(False)
        self.filename.setVisible(False)
        self.loadcamera.setVisible(True)
        self.loadvideo.setVisible(True)
        self.Startlabel.setVisible(True)
        self.Start.setVisible(False)
        self.prob_label.setVisible(False)
        self.Problem.setVisible(False)
        self.Record_area.setVisible(False)
        self.Record.setVisible(False)
        self.camera_start.setVisible(False)
        self.Clear.setVisible(False)
        _translate = QtCore.QCoreApplication.translate
        self.camera_start.setText(_translate("Dialog", "Start"))
        self.Problem.clear() 
        self.Record.clear()
        self.A01.setStyleSheet('''color:#c3c3c3''')
        self.A02.setStyleSheet('''color:#c3c3c3''')
        self.A04.setStyleSheet('''color:#c3c3c3''')
        self.A05.setStyleSheet('''color:#c3c3c3''')
        self.A06.setStyleSheet('''color:#c3c3c3''')
        self.A09.setStyleSheet('''color:#c3c3c3''')
        self.A12.setStyleSheet('''color:#c3c3c3''')
        self.A15.setStyleSheet('''color:#c3c3c3''')
        self.A17.setStyleSheet('''color:#c3c3c3''')
        self.A20.setStyleSheet('''color:#c3c3c3''')
        self.A25.setStyleSheet('''color:#c3c3c3''')
        self.A26.setStyleSheet('''color:#c3c3c3''')
        self.Happly_label.setStyleSheet('''color:#c3c3c3''')
        self.Angry_label.setStyleSheet('''color:#c3c3c3''')
        self.DIsgust_label.setStyleSheet('''color:#c3c3c3''')
        self.Fear_label.setStyleSheet('''color:#c3c3c3''')
        self.Sad_label.setStyleSheet('''color:#c3c3c3''')
        self.Neutral_label.setStyleSheet('''color:#c3c3c3''')
        self.Surprise_label.setStyleSheet('''color:#c3c3c3''')
        self.color = (0, 255, 0)
        self.truth_lie.setText('Lie_truth')
        # self.truth_lie.setStyleSheet('''QPushButton{background:##ff70ff;border-radius:5px;}''')
        self.videoprogress.setValue(0)
        self.frame_embed_list = []
        self.frame_emb_AU = []
        self.userface = []
        self.countframe = 0
        self.index = 0
        self.len_bbox = 1
        self.Original.setPixmap(QPixmap(""))
        # self.Facealignment.setPixmap(QPixmap(""))
        # self.Landmark.setPixmap(QPixmap(""))
        self.filename.setText('')
        self.Startlabel.setText('Press the button to upload a video or activate camera')
        # self.Facedetection.setPixmap(QPixmap(""))
        self.timer.stop()
        if self.cap != None:
            self.cap.release()
            self.cap = None
if __name__=='__main__':
    app = QApplication(sys.argv)
    window = lie_GUI(args)
    window.show()
    sys.exit(app.exec_())