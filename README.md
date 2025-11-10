# Face + Emotion + Micro-Expression → Lie Detection Pipeline

This project ties together several computer-vision models to estimate whether a person is being truthful or deceptive on a frame-by-frame basis.

The system runs through four stages:

1. Face detection
2. Emotion feature extraction
3. Micro-expression feature extraction
4. SVM-based truth vs. lie prediction

The final output is a probability-based decision for each frame.

---

## 1) Face Detection — RetinaFace

**File**
`model/Facedetection/RetinaFace/RetinaFaceDetection.py`

**Weights**
`model/Facedetection/RetinaFace/weights/mobilenet0.25_Final.pth`

**What it does**

* Detects faces
* Returns bounding boxes and landmarks
* Provides aligned face crops for downstream models

**Backbone**
MobileNet-0.25

---

## 2) Emotion Recognition — Self-Relation Attention Model

**File**
`model/Emotion/lie_emotion_process.py`

**Checkpoint**
`model/Emotion/model112/self_relation-attention_AFEW_better_46.0733_41.2759_12.tar`

**What it does**

* Takes the aligned face
* Generates an emotional feature embedding
* Predicts one of:

```
['Happy', 'Angry', 'Disgust', 'Fear', 'Sad', 'Neutral', 'Surprise']
```

**Architecture**
ResNet with a custom self-relation attention module

**Why it matters**
The emotional embedding becomes part of the final deception classifier.

---

## 3) Micro-Expression Extraction

**File**
`model/action_v4_L12_BCE_MLSM/lie_action_process.py`

**Config**
`model/action_v4_L12_BCE_MLSM/config.py`

**What it does**

* Looks for subtle, short-lived facial movements
  Examples:

  * brief brow lift
  * jaw drop
  * slight lip corner pull
* Produces:

  * micro-expression predictions
  * a compact embedding representing these cues

**Architecture**
Modified ResNet backbone

**Why it matters**
These micro-expression features carry valuable signal about emotion leakage, which helps the SVM decide truth vs. deception.

---

## 4) Lie vs. Truth Classification — SVM

**File**
`model/SVM_model/se_res50+EU_v2/se_res50+EU/split_svc_AUC0.872.joblib`

**Model type**
`sklearn.svm.SVC(probability=True)`

**What it does**

* Takes two embeddings:

  * micro-expression representation
  * emotion representation
* Outputs:

  * predicted class: Truth or Lie
  * confidence score

This is the final decision stage used by both the backend and the GUI.

---

## End-to-End Flow

1. Detect and crop face with RetinaFace
2. Extract emotional features
3. Extract micro-expression features
4. Combine both embeddings
5. SVM predicts truth or lie with a probability score

---

## Output Format

Each processed frame can look something like this:

```json
{
  "truth": true,
  "probability": 0.78,
  "emotion": "Neutral",
  "microexpressions": [...],
  "embeddings": {
    "emotion": [...],
    "microexpression": [...]
  }
}
```

---

## Folder Layout

```
model/
│
├── Facedetection/
│   └── RetinaFace/
│       └── ...
│
├── Emotion/
│   └── lie_emotion_process.py
│
├── action_v4_L12_BCE_MLSM/
│   └── lie_action_process.py
│
└── SVM_model/
    └── se_res50+EU_v2/
        └── split_svc_AUC0.872.joblib
```

---

## Notes

* Faces are aligned before running the recognition models
* Emotion + micro-expression features form the core input to the final classifier
* The SVM is used only for inference; training is performed beforehand

---

