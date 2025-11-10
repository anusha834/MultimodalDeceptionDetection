"""
Flask Backend API for Deception Detection (REPLACEMENT)
Usage: python backend.py
Serves:
 - GET  /api/health
 - POST /api/analyze_frame  { frame: base64_image }
 - POST /api/analyze_video  (multipart file 'video')
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
from joblib import load
import base64
from io import BytesIO
from PIL import Image
import os
import sys
import argparse
import skimage
from skimage import img_as_ubyte

# Make sure repo root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local code (these modules must be present in repo)
from model.Facedetection.RetinaFace.RetinaFaceDetection import retina_face
import model.Emotion.lie_emotion_process as emotion
import model.action_v4_L12_BCE_MLSM.lie_action_process as action
from model.action_v4_L12_BCE_MLSM.config import Config
from model.Facedetection.utils import align_face

app = Flask(__name__)
CORS(app)

# Globals for models
Retina = None
Emotion_class = None
Action_class = None
SVM_model = None
MODEL_ARGS = None

# AU names (same order your GUI expects)
AU_NAMES = ['Inner brow raiser','Outer brow raiser','Brow lower','Upper Lid Raiser',
            'Cheek raiser','Nose wrinkle','Lip corner puller','Lip corner depressor',
            'Chin raiser','Lip Stretcher','Lips part','Jaw drop']

EMOTION_LABELS = ['Happy','Angry','Disgust','Fear','Sad','Neutral','Surprise']

def initialize_models():
    global Retina, Emotion_class, Action_class, SVM_model, MODEL_ARGS
    parser = argparse.ArgumentParser()
    # RetinaFace args (matching your lie_GUI defaults)
    parser.add_argument('--gpu_num', default="0", type=str)
    parser.add_argument('--vis_thres', default=0.6, type=float)
    parser.add_argument('--network', default='mobile0.25')
    parser.add_argument('--trained_model', default='./model/Facedetection/RetinaFace/weights/mobilenet0.25_Final.pth', type=str)
    parser.add_argument('--cpu', action='store_true', help='use cpu mode for RetinaFace')
    parser.add_argument('--confidence_threshold', default=0.6, type=float)
    parser.add_argument('--top_k', default=5000, type=int)
    parser.add_argument('--keep_top_k', default=750, type=int)
    parser.add_argument('--nms_threshold', default=0.4, type=float)
    parser.add_argument('--save_image', action='store_false', help='disable image saving in Flask backend')


    # Emotion args
    parser.add_argument('--at_type', default=1, type=int)
    parser.add_argument('--preTrain_path', default='./model/Emotion/model112/self_relation-attention_AFEW_better_46.0733_41.2759_12.tar')
    parser.add_argument('-c', '--config', type=str, default='./model/Landmark/configs/mb1_120x120.yml')
    parser.add_argument('--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d', choices=['2d','3d'])
    args = parser.parse_args([])  # avoid reading actual CLI args
    MODEL_ARGS = args
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

    print("Initializing models (this may take a while)...")
    # RetinaFace
    Retina = retina_face(crop_size=224, args=args)
    print("Loaded RetinaFace.")

    # Emotion model
    Emotion_class = emotion.Emotion_FAN(args=args)
    print("Loaded Emotion model.")

    # Action AU model
    Action_class = action.Action_Resnet(args=Config())
    print("Loaded Action (AU) model.")

    # ✅ Updated SVM loading
    svm_path = './model/SVM_model/se_res50+EU_v2/se_res50+EU/split_svc_AUC0.872.joblib'
    if not os.path.exists(svm_path):
        print(f"[WARN] Preferred SVM model not found at {svm_path}, falling back to older version.")
        svm_path = './model/SVM_model/se_res50+EU/split_svc_acc0.720_AUC0.828.joblib'

    SVM_model = load(svm_path)
    print(f"Loaded SVM model: {svm_path}")

    # Quick introspection
    print("SVM classes:", getattr(SVM_model, "classes_", None))
    print("SVM predict_proba:", hasattr(SVM_model, "predict_proba"))
    print("Initialization complete.")


def safe_tensor_to_numpy(x):
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return np.array(x)

def analyze_all_faces_in_frame(frame_bgr):
    """
    Core logic: detect faces, align each face, run Action & Emotion models,
    construct feature vector and SVM prediction/prob for each face.
    Returns list of per-face dictionaries.
    """
    # Keep original copy for bbox coordinates (we won't resize before detection to preserve coords)
    try:
        image = skimage.img_as_float(frame_bgr).astype(np.float32)
        frame_ubyte = img_as_ubyte(image)
    except Exception:
        # fallback
        frame_ubyte = frame_bgr

    result = Retina.detect_face(frame_ubyte)
    # result format: either 4 items (no face_list) or 5 items (with face_list)
    if len(result) == 4:
        img_raw, output_raw, output_points, bbox = result
        face_list = [bbox] if bbox is not None else []
    else:
        img_raw, output_raw, output_points, bbox, face_list = result

    # Normalize returned structures
    if face_list is None:
        face_list = []
    if bbox is None:
        bbox = []

    # If bbox is a single box and not a nested list, wrap it to list so iteration is uniform
    try:
        # bbox sometimes provided as ndarray shape (n,4)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and (len(face_list) == 0):
            bbox = [bbox]
    except Exception:
        pass

    faces_out = []
    if len(bbox) == 0:
        return faces_out  # no faces

    # Loop over detected faces
    for idx in range(len(bbox)):
        try:
            # Align the face using corresponding output_points (landmarks)
            pts = output_points[idx] if isinstance(output_points, (list, tuple, np.ndarray)) and len(output_points) > idx else output_points[0]
            out_raw = align_face(output_raw, pts, crop_size_h=112, crop_size_w=112)
            out_raw = cv2.resize(out_raw, (224,224), interpolation=cv2.INTER_AREA)
        except Exception as e:
            # alignment failed -> skip this face
            print("Face alignment failed for idx", idx, "err:", e)
            continue

        # Action model: returns AU logits/activations and embedding
        logps, emb = Action_class._pred(out_raw, Config)
        logps_np = safe_tensor_to_numpy(logps)
        emb_np = safe_tensor_to_numpy(emb)
        # shape adjustments
        if logps_np.ndim == 2 and logps_np.shape[0] == 1:
            logps_1d = logps_np[0]
        else:
            logps_1d = logps_np.reshape(-1)
        # AU binarization as in original app
        AU_binary = [1 if float(x) >= 0.01 else 0 for x in logps_1d]

        # Emotion
        pred_score, self_embedding, relation_embedding = Emotion_class.validate([out_raw])
        pred_score_np = safe_tensor_to_numpy(pred_score)
        if pred_score_np.ndim == 2 and pred_score_np.shape[0] == 1:
            pred_score_1d = pred_score_np[0]
        else:
            pred_score_1d = pred_score_np.reshape(-1)
        emotion_idx = int(np.argmax(pred_score_1d))
        emotion_label = EMOTION_LABELS[emotion_idx]

        # Prepare embeddings for SVM
        # emb may be (1, N) or (N,)
        frame_emb_AU = emb_np.reshape(1, -1).astype(np.float32)
        rel_emb_np = safe_tensor_to_numpy(relation_embedding).reshape(1, -1).astype(np.float32)

        # Concatenate features for SVM exactly as desktop app
        try:
            feature = np.concatenate((frame_emb_AU, rel_emb_np), axis=1)
        except Exception:
            feature = np.hstack([frame_emb_AU.reshape(1, -1), rel_emb_np.reshape(1, -1)])
        feature_sklearn = feature.astype(np.float64)

        # SVM prediction & probability (robust)
        try:
            results_arr = SVM_model.predict(feature_sklearn)
            results_scalar = int(np.asarray(results_arr).reshape(-1)[0])
        except Exception:
            results_scalar = 0
        prob = 0.0
        try:
            if hasattr(SVM_model, "predict_proba"):
                probs = SVM_model.predict_proba(feature_sklearn)
                try:
                    idx1 = list(SVM_model.classes_).index(1)
                    prob = float(probs[0, idx1])
                except:
                    prob = float(np.max(probs[0]))
            elif hasattr(SVM_model, "decision_function"):
                df = SVM_model.decision_function(feature_sklearn)
                df0 = float(np.asarray(df).reshape(-1)[0])
                prob = 1.0 / (1.0 + np.exp(-df0))
            else:
                prob = 1.0 if results_scalar == 1 else 0.0
        except Exception:
            prob = 0.0

        # Convert bbox element to plain python list (x1,y1,x2,y2)
        try:
            if isinstance(bbox[idx], (list, tuple)):
                bbox_list = [int(float(x)) for x in bbox[idx]]
            elif hasattr(bbox[idx], 'tolist'):
                bbox_list = [int(float(x)) for x in bbox[idx].tolist()]
            else:
                # fallback: zeros
                bbox_list = [0,0,0,0]
        except Exception:
            bbox_list = [0,0,0,0]

        active_au_names = [AU_NAMES[i] for i, val in enumerate(AU_binary) if val == 1]

        face_dict = {
            'bbox': bbox_list,
            'action_units': AU_binary,
            'active_action_units': active_au_names,
            'emotion_scores': [float(x) for x in pred_score_1d.tolist()],
            'emotion': emotion_label,
            'prediction': 'Deception' if results_scalar == 1 else 'Truth',
            'deception_probability': float(prob * 100.0),
            'is_deception': bool(results_scalar == 1)
        }
        faces_out.append(face_dict)
    return faces_out

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': Retina is not None})

@app.route('/api/analyze_frame', methods=['POST'])
def analyze_frame():
    try:
        data = request.json
        if data is None or 'frame' not in data:
            return jsonify({'error': 'No frame provided'}), 400
        # base64 decode (data URL or raw base64)
        b64 = data['frame']
        if b64.startswith('data:'):
            b64 = b64.split(',', 1)[1]
        img_data = base64.b64decode(b64)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Optionally resize for consistent processing (RetinaFace handles arbitrary sizes)
        # Keep native size to preserve bbox consistency
        faces = analyze_all_faces_in_frame(frame)
        if len(faces) == 0:
            return jsonify({'face_detected': False, 'faces': []})
        return jsonify({'face_detected': True, 'faces': faces})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    video_file = request.files['video']
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tmp.close()
    path = tmp.name
    try:
        video_file.save(path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return jsonify({'error': 'Failed to open video file'}), 400
        results = []
        frame_count = 0
        max_frames = 300  # safety cap; front-end already samples
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # sample every N frames (front-end also samples); here we'll capture each but only process a subset
            if frame_count % 30 == 0:
                faces = analyze_all_faces_in_frame(frame)
                results.append({'frame_number': frame_count, 'faces': faces})
            frame_count += 1
        cap.release()
        try:
            os.remove(path)
        except:
            pass
        # build summary identical in spirit to desktop app
        flat_faces = [f for r in results for f in r['faces']]
        if len(flat_faces) == 0:
            return jsonify({'error': 'No faces detected in video'}), 400
        deception_frames = sum(1 for f in flat_faces if f.get('is_deception', False))
        total_frames = len(flat_faces)
        avg_probability = float(np.mean([f.get('deception_probability', 0.0) for f in flat_faces]))
        overall = 'Deception Detected' if avg_probability > 50 else 'Truthful'
        return jsonify({
            'frame_results': results,
            'summary': {
                'total_frames_analyzed': total_frames,
                'deception_frames': deception_frames,
                'truth_frames': total_frames - deception_frames,
                'average_deception_probability': avg_probability,
                'overall_assessment': overall
            }
        })
    except Exception as e:
        try:
            os.remove(path)
        except:
            pass
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_models()
    # Run on 0.0.0.0 so local web html can call it
    app.run(host='0.0.0.0', port=5000, debug=False)
