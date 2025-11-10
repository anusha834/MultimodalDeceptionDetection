---

# Face + Emotion + AU → Lie Detection Pipeline

This project pulls together a full vision pipeline to decide whether a person is being truthful or deceptive on a **frame-by-frame** basis.
It runs four stages:

1. Face Detection →
2. Emotion Features →
3. Action Unit Features →
4. SVM Classification
   …then outputs **Truth / Lie** with probability.

Below is a simple breakdown of how everything fits together.

---

## ✅ 1) Face Detection — RetinaFace

**File**
`model/Facedetection/RetinaFace/RetinaFaceDetection.py`

**Weights**
`model/Facedetection/RetinaFace/weights/mobilenet0.25_Final.pth`

**What it does**

* Finds faces
* Returns bounding boxes + 5 facial landmarks
* Used to crop + align the face before feature extraction

**Backbone**
MobileNet-0.25

---

## ✅ 2) Emotion Recognition — Self-Relation Attention

**File**
`model/Emotion/lie_emotion_process.py`

**Checkpoint**
`model/Emotion/model112/self_relation-attention_AFEW_better_46.0733_41.2759_12.tar`

**What it does**

* Takes cropped face
* Produces an emotional embedding
* Predicts one of:

```
['Happy', 'Angry', 'Disgust', 'Fear', 'Sad', 'Neutral', 'Surprise']
```

**Architecture**
ResNet + custom self-relation attention module

**Used for**

* Feature embedding passed into final SVM

---

## ✅ 3) Action Unit (AU) Extraction — Action_ResNet

**File**
`model/action_v4_L12_BCE_MLSM/lie_action_process.py`

**Config**
`model/action_v4_L12_BCE_MLSM/config.py`

**What it does**

* Extracts 12 facial Action Units (AUs)
  e.g. brow raise, lip corner puller, jaw drop
* Produces:

  * AU predictions
  * High-level AU embedding

**Architecture**
Modified SE-ResNet

**Used for**

* Feature embedding passed into final SVM

---

## ✅ 4) Lie / Truth Classification — SVM

**File**
`model/SVM_model/se_res50+EU_v2/se_res50+EU/split_svc_AUC0.872.joblib`

**Model type**
`sklearn.svm.SVC(probability=True)`

**What it does**

* Inputs:

  * AU embedding
  * Emotion embedding
* Outputs:

  * Truth / Lie
  * Probability score

This is the final decision for each frame.
Used by both the GUI and backend.

---

## 📦 End-to-End Flow

1. Detect + crop face → RetinaFace
2. Extract emotion embedding → Self-Relation attention model
3. Extract AU embedding → Action ResNet
4. Concatenate embeddings
5. Feed to SVM → Truth / Lie + probability

---

## Output

For each frame:

```json
{
  "truth": true,
  "probability": 0.78,
  "emotion": "Neutral",
  "AU_features": [...],
  "embeddings": {
    "emotion": [...],
    "AU": [...]
  }
}
```

---

## Folder Summary

```
model/
│
├── Facedetection/
│   └── RetinaFace/...
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

* Face alignment is handled before feeding into downstream models
* The SVM is trained offline; only inference runs here
* Emotion + AU embeddings are the main signal for the final classifier

---
