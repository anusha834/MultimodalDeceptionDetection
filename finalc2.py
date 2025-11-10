import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import tempfile
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import warnings
from typing import Dict, List, Tuple, Optional
import re


TRANSFORMERS_AVAILABLE = False
LIBROSA_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False
PYAUDIO_AVAILABLE = False
SPEECH_RECOGNITION_AVAILABLE = False
SKLEARN_AVAILABLE = False
WHISPER_DIRECT_AVAILABLE = False


try:
    from transformers import (
        AutoTokenizer, AutoModel, 
        Wav2Vec2Processor, Wav2Vec2Model,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
    print("✓ Transformers loaded successfully")
except ImportError as e:
    print(f"⚠ Transformers not available: {e}")
    print("Advanced AI features disabled. Fix with:")
    print("  pip uninstall protobuf tensorflow transformers")
    print("  pip install protobuf==3.20.3 tensorflow==2.13.0 transformers")


try:
    import librosa
    LIBROSA_AVAILABLE = True
    print("✓ Librosa loaded successfully")
except ImportError:
    print("⚠ Librosa not available - advanced audio analysis disabled")
    print("Install with: pip install librosa")


try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe loaded successfully")
except ImportError:
    print("⚠ MediaPipe not available - facial analysis disabled")
    print("Install with: pip install mediapipe")


try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
    print("✓ PyAudio loaded successfully")
except ImportError:
    print("⚠ PyAudio not available - audio recording disabled")
    print("Install with: pip install pyaudio")


try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
    print("✓ SpeechRecognition loaded successfully")
except ImportError:
    print("⚠ SpeechRecognition not available - transcription disabled")
    print("Install with: pip install speechrecognition")


try:
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    SKLEARN_AVAILABLE = True
    print("✓ Scikit-learn loaded successfully")
except ImportError:
    print("⚠ Scikit-learn not available - some analysis features disabled")
    print("Install with: pip install scikit-learn pandas")


try:
    import whisper
    WHISPER_DIRECT_AVAILABLE = True
    print("✓ OpenAI Whisper (direct) loaded successfully")
except ImportError:
    print("⚠ Direct Whisper not available - install with: pip install openai-whisper")

warnings.filterwarnings('ignore')


class CameraRecorder:
    """Handles camera recording with audio"""
    
    def __init__(self):
        self.is_recording = False
        self.video_capture = None
        self.video_writer = None
        self.audio_frames = []
        self.audio = None
        

        if PYAUDIO_AVAILABLE:
            self.audio_format = pyaudio.paInt16
            self.channels = 1
            self.rate = 44100
            self.chunk = 1024
        
        self.temp_dir = tempfile.mkdtemp()
        
    def start_recording(self, output_path: str = None):
        """Start recording video and audio"""
        if self.is_recording:
            return False
            
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.temp_dir, f"recording_{timestamp}.mp4")
        
        self.output_path = output_path
        self.is_recording = True
        

        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            self.is_recording = False
            return False
            

        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)
        

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, 30.0, (640, 480)
        )
        

        if PYAUDIO_AVAILABLE:
            try:
                self.audio = pyaudio.PyAudio()
                self.audio_stream = self.audio.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    frames_per_buffer=self.chunk
                )
                self.audio_frames = []
            except Exception as e:
                print(f"Audio initialization failed: {e}")
                self.audio = None
        

        self.video_thread = threading.Thread(target=self._record_video)
        if PYAUDIO_AVAILABLE and self.audio:
            self.audio_thread = threading.Thread(target=self._record_audio)
            self.audio_thread.start()
        
        self.video_thread.start()
        
        return True
    
    def stop_recording(self):
        """Stop recording and save files"""
        if not self.is_recording:
            return None, None
            
        self.is_recording = False
        

        if hasattr(self, 'video_thread'):
            self.video_thread.join()
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
        

        if self.video_capture:
            self.video_capture.release()
        if self.video_writer:
            self.video_writer.release()
        

        audio_path = None
        if PYAUDIO_AVAILABLE and self.audio_frames:
            audio_path = self.output_path.replace('.mp4', '.wav')
            self._save_audio(audio_path)
        

        if hasattr(self, 'audio_stream') and self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.audio:
            self.audio.terminate()
        
        return self.output_path, audio_path
    
    def _record_video(self):
        """Video recording thread"""
        while self.is_recording:
            ret, frame = self.video_capture.read()
            if ret:
                self.video_writer.write(frame)

                cv2.imshow('Recording... (Press Q to stop)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_recording = False
                    break
            else:
                break
        cv2.destroyAllWindows()
    
    def _record_audio(self):
        """Audio recording thread"""
        if not self.audio:
            return
            
        while self.is_recording:
            try:
                data = self.audio_stream.read(self.chunk, exception_on_overflow=False)
                self.audio_frames.append(data)
            except Exception as e:
                print(f"Audio recording error: {e}")
                break
    
    def _save_audio(self, audio_path: str):
        """Save recorded audio to file"""
        try:
            wf = wave.open(audio_path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()
        except Exception as e:
            print(f"Error saving audio: {e}")


class EnhancedSpeechToTextConverter:
    """Enhanced speech-to-text that preserves filler words and disfluencies"""
    
    def __init__(self):
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
        

        if WHISPER_DIRECT_AVAILABLE:
            try:

                self.whisper_direct = whisper.load_model("small")
                self.use_direct_whisper = True
            except Exception as e:
                print(f"Failed to load direct Whisper: {e}")
                self.whisper_direct = None
                self.use_direct_whisper = False
        else:
            self.whisper_direct = None
            self.use_direct_whisper = False
        

        self.use_transformers_whisper = False
        if TRANSFORMERS_AVAILABLE and not self.use_direct_whisper:
            try:
                self.whisper_model = pipeline(
                    "automatic-speech-recognition", 
                    model="openai/whisper-small",
                    return_timestamps=True
                )
                self.use_transformers_whisper = True
                print("✓ Transformers Whisper model loaded")
            except:
                print("Transformers Whisper model not available")
    
    def convert_audio_to_text_with_fillers(self, audio_path: str) -> Tuple[str, Dict]:
        """Convert audio to text while preserving filler words and detecting disfluencies"""
        

        if self.use_direct_whisper:
            text = self._direct_whisper_transcribe(audio_path)
        elif self.use_transformers_whisper:
            text = self._transformers_whisper_transcribe(audio_path)
        elif SPEECH_RECOGNITION_AVAILABLE:
            text = self._enhanced_google_transcribe(audio_path)
        else:
            text = "Transcription not available"
        

        disfluency_features = self._detect_audio_disfluencies(audio_path)
        

        enhanced_text = self._enhance_text_with_audio_disfluencies(text, disfluency_features)
        
        return enhanced_text, disfluency_features
    
    def _direct_whisper_transcribe(self, audio_path: str) -> str:
        """Use direct OpenAI Whisper - best for preserving natural speech patterns"""
        try:

            result = self.whisper_direct.transcribe(
                audio_path,
                language='english',
                task='transcribe',
                verbose=False,
                word_timestamps=True,
                initial_prompt="Include all speech sounds like um, uh, ah, like, you know, and other natural speech patterns."
            )
            

            text = result["text"]
            

            text = self._post_process_for_natural_speech(text)
            
            print(f"Direct Whisper transcription: {text}")
            return text
            
        except Exception as e:
            print(f"Direct Whisper transcription error: {e}")
            return self._fallback_transcription(audio_path)
    
    def _transformers_whisper_transcribe(self, audio_path: str) -> str:
        """Use transformers Whisper pipeline"""
        try:
            result = self.whisper_model(audio_path)
            text = result["text"] if isinstance(result, dict) else str(result)
            
            text = self._post_process_for_natural_speech(text)
            print(f"Transformers Whisper transcription: {text}")
            return text
            
        except Exception as e:
            print(f"Transformers Whisper error: {e}")
            return self._fallback_transcription(audio_path)
    
    def _enhanced_google_transcribe(self, audio_path: str) -> str:
        """Enhanced Google Speech Recognition with less filtering"""
        try:
            with sr.AudioFile(audio_path) as source:

                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.recognizer.energy_threshold = 200  # Lower threshold to catch quieter sounds
                self.recognizer.dynamic_energy_threshold = True
                
                audio = self.recognizer.record(source)
                

                try:

                    result = self.recognizer.recognize_google(
                        audio, 
                        show_all=True,
                        language='en-US'
                    )
                    
                    if result and 'alternative' in result:

                        text = result['alternative'][0]['transcript']
                    else:
                        text = "No speech detected"
                        
                except:

                    text = self.recognizer.recognize_google(audio, language='en-US')
                
                text = self._post_process_for_natural_speech(text)
                print(f"Enhanced Google transcription: {text}")
                return text
                
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Speech recognition error: {e}"
        except Exception as e:
            print(f"Enhanced Google transcription error: {e}")
            return "Transcription failed"
    
    def _fallback_transcription(self, audio_path: str) -> str:
        """Fallback transcription method"""
        if SPEECH_RECOGNITION_AVAILABLE:
            return self._enhanced_google_transcribe(audio_path)
        else:
            return "No transcription method available"
    
    def _post_process_for_natural_speech(self, text: str) -> str:
        """Post-process transcription to restore natural speech patterns"""
        

        corrections = {

            r'\b(I|we|you|they)\s+(think|mean|guess|believe)\b': r'\1, \2,',  # Add hesitation
            r'\b(well|so|like|just)\s+([a-z])': r'\1, \2',  # Add pause markers
            r'\b(maybe|perhaps|possibly|probably)\b': r'well, \1',  # Add hesitation before uncertainty
        }
        
        processed_text = text
        for pattern, replacement in corrections.items():
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
        
        return processed_text
    
    def _detect_audio_disfluencies(self, audio_path: str) -> Dict:
        """Detect disfluencies directly from audio signal"""
        disfluencies = {
            'short_segments': 0,
            'hesitation_pauses': 0,
            'repeated_patterns': 0,
            'filled_pauses': 0,
            'total_disfluencies': 0
        }
        
        if not LIBROSA_AVAILABLE:
            return disfluencies
        
        try:

            audio, sr = librosa.load(audio_path, sr=16000)
            

            intervals = librosa.effects.split(audio, top_db=20, frame_length=2048, hop_length=512)
            

            for start, end in intervals:
                segment_length = (end - start) / sr
                segment_audio = audio[start:end]
                

                if 0.1 <= segment_length <= 0.8:
                    disfluencies['short_segments'] += 1
                    

                    if self._is_filled_pause(segment_audio, sr):
                        disfluencies['filled_pauses'] += 1
                

                if segment_length < 0.1:
                    disfluencies['hesitation_pauses'] += 1
            

            disfluencies['repeated_patterns'] = self._detect_repetitions(audio, sr)
            

            disfluencies['total_disfluencies'] = (
                disfluencies['short_segments'] + 
                disfluencies['hesitation_pauses'] + 
                disfluencies['repeated_patterns'] +
                disfluencies['filled_pauses']
            )
            
        except Exception as e:
            print(f"Error detecting audio disfluencies: {e}")
        
        return disfluencies
    
    def _is_filled_pause(self, audio_segment: np.ndarray, sr: int) -> bool:
        """Determine if audio segment is likely a filled pause (um, uh, er)"""
        try:




            

            pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=sr, threshold=0.1)
            

            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if not pitch_values:
                return False
            



            mean_pitch = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            
            return (80 <= mean_pitch <= 200) and (pitch_std < 30)
            
        except Exception:
            return False
    
    def _detect_repetitions(self, audio: np.ndarray, sr: int) -> int:
        """Detect repeated patterns in audio (word repetition, stuttering)"""
        try:

            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            

            repetitions = 0
            window_size = 20  # frames
            
            for i in range(len(mfccs[0]) - window_size * 2):
                segment1 = mfccs[:, i:i+window_size]
                segment2 = mfccs[:, i+window_size:i+window_size*2]
                

                correlation = np.corrcoef(segment1.flatten(), segment2.flatten())[0, 1]
                
                if correlation > 0.8:  # High similarity indicates repetition
                    repetitions += 1
            
            return repetitions
            
        except Exception:
            return 0
    
    def _enhance_text_with_audio_disfluencies(self, text: str, disfluency_features: Dict) -> str:
        """Enhance transcribed text with disfluencies detected from audio"""
        
        enhanced_text = text
        

        if disfluency_features['filled_pauses'] > 0:

            words = enhanced_text.split()
            

            uncertainty_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'probably', 'think', 'guess']
            for i, word in enumerate(words):
                if word.lower() in uncertainty_words and i > 0:
                    words[i] = f"um, {word}"
            
            enhanced_text = ' '.join(words)
        
        if disfluency_features['hesitation_pauses'] > 2:

            enhanced_text = re.sub(r'\b(so|well|like)\b', r'uh, \1', enhanced_text, count=2)
        
        return enhanced_text


class FacialAnalyzer:
    """Extract facial micro-expressions and eye gaze features"""
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            return
            
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        

        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.mouth_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        
    def extract_features(self, video_path: str) -> Dict:
        """Extract temporal facial features from video"""
        if not MEDIAPIPE_AVAILABLE:
            return {
                'blink_rate': 0,
                'eye_aspect_ratios': [0],
                'mouth_movement': [0],
                'micro_expressions': [0],
                'gaze_direction': [0],
                'timestamps': [0]
            }
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        features = {
            'blink_rate': 0,
            'eye_aspect_ratios': [],
            'mouth_movement': [],
            'micro_expressions': [],
            'gaze_direction': [],
            'timestamps': []
        }
        
        frame_count = 0
        prev_landmarks = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            timestamp = frame_count / fps if fps > 0 else frame_count
            features['timestamps'].append(timestamp)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                

                ear = self._calculate_eye_aspect_ratio(landmarks)
                features['eye_aspect_ratios'].append(ear)
                

                mouth_movement = self._calculate_mouth_movement(landmarks, prev_landmarks)
                features['mouth_movement'].append(mouth_movement)
                

                gaze = self._estimate_gaze_direction(landmarks)
                features['gaze_direction'].append(gaze)
                

                micro_expr = self._detect_micro_expression(landmarks, prev_landmarks)
                features['micro_expressions'].append(micro_expr)
                
                prev_landmarks = landmarks
            else:

                for key in ['eye_aspect_ratios', 'mouth_movement', 'gaze_direction', 'micro_expressions']:
                    features[key].append(0.0)
            
            frame_count += 1
        
        cap.release()
        

        if features['eye_aspect_ratios']:
            blinks = self._detect_blinks(features['eye_aspect_ratios'])
            total_time = frame_count / fps if fps > 0 else 1
            features['blink_rate'] = len(blinks) / total_time * 60
        
        return features
    
    def _calculate_eye_aspect_ratio(self, landmarks) -> float:
        """Calculate Eye Aspect Ratio for blink detection"""
        def distance(p1, p2):
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        try:

            left_ear = (distance(landmarks.landmark[159], landmarks.landmark[145]) + 
                       distance(landmarks.landmark[158], landmarks.landmark[153])) / \
                      (2.0 * distance(landmarks.landmark[33], landmarks.landmark[133]))
            

            right_ear = (distance(landmarks.landmark[386], landmarks.landmark[374]) + 
                        distance(landmarks.landmark[385], landmarks.landmark[380])) / \
                       (2.0 * distance(landmarks.landmark[362], landmarks.landmark[263]))
            
            return (left_ear + right_ear) / 2.0
        except:
            return 0.25  # Default value
    
    def _calculate_mouth_movement(self, landmarks, prev_landmarks) -> float:
        """Calculate mouth movement between frames"""
        if prev_landmarks is None:
            return 0.0
        
        try:
            movement = 0.0
            for idx in self.mouth_indices:
                curr = landmarks.landmark[idx]
                prev = prev_landmarks.landmark[idx]
                movement += np.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2)
            
            return movement / len(self.mouth_indices)
        except:
            return 0.0
    
    def _estimate_gaze_direction(self, landmarks) -> Tuple[float, float]:
        """Estimate gaze direction (simplified)"""
        try:
            left_eye_center = np.mean([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
                                      for i in [33, 133]], axis=0)
            right_eye_center = np.mean([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
                                       for i in [362, 263]], axis=0)
            

            gaze_x = (left_eye_center[0] + right_eye_center[0]) / 2
            gaze_y = (left_eye_center[1] + right_eye_center[1]) / 2
            
            return gaze_x, gaze_y
        except:
            return 0.5, 0.5
    
    def _detect_micro_expression(self, landmarks, prev_landmarks) -> float:
        """Detect micro-expressions based on landmark movement"""
        if prev_landmarks is None:
            return 0.0
        
        try:

            key_regions = self.mouth_indices + [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
            
            movement = 0.0
            for idx in key_regions:
                curr = landmarks.landmark[idx]
                prev = prev_landmarks.landmark[idx]
                movement += np.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2)
            
            return movement / len(key_regions)
        except:
            return 0.0
    
    def _detect_blinks(self, ear_values: List[float], threshold: float = 0.25) -> List[int]:
        """Detect blinks in EAR sequence"""
        blinks = []
        for i in range(1, len(ear_values)):
            if ear_values[i-1] > threshold and ear_values[i] <= threshold:
                blinks.append(i)
        return blinks


class AudioAnalyzer:
    """Extract speech and acoustic features"""
    
    def __init__(self):
        self.features_available = LIBROSA_AVAILABLE
        

        if TRANSFORMERS_AVAILABLE:
            try:
                self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            except:
                print("Wav2Vec2 model not available")
                self.wav2vec_processor = None
                self.wav2vec_model = None
            

            try:
                self.emotion_classifier = pipeline("audio-classification", 
                                                 model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
            except:
                print("Emotion classifier not available")
                self.emotion_classifier = None
        else:
            self.wav2vec_processor = None
            self.wav2vec_model = None
            self.emotion_classifier = None
    
    def extract_features(self, audio_path: str) -> Dict:
        """Extract comprehensive audio features"""
        if not self.features_available:
            return self._get_default_features()
        

        try:
            audio, sr = librosa.load(audio_path, sr=16000)
        except:
            print(f"Error loading audio file: {audio_path}")
            return self._get_default_features()
        
        features = {}
        

        features.update(self._extract_prosodic_features(audio, sr))
        

        if self.wav2vec_model is not None:
            features.update(self._extract_wav2vec_features(audio))
        

        if self.emotion_classifier is not None:
            features.update(self._extract_emotion_features(audio_path))
        

        features.update(self._extract_voice_quality_features(audio, sr))
        
        return features
    
    def _get_default_features(self) -> Dict:
        """Return default features if audio processing fails"""
        return {
            'pitch_mean': 0, 'pitch_std': 0, 'pitch_range': 0,
            'speech_rate': 0, 'pause_ratio': 0, 'energy_mean': 0, 'energy_std': 0,
            'spectral_centroid_mean': 0, 'spectral_centroid_std': 0,
            'spectral_rolloff_mean': 0, 'zcr_mean': 0, 'zcr_std': 0
        }
    
    def _extract_prosodic_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract prosodic features (pitch, rhythm, etc.)"""
        features = {}
        
        try:

            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.1)
            pitch_values = []
            
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            else:
                features['pitch_mean'] = features['pitch_std'] = features['pitch_range'] = 0
            

            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            features['speech_rate'] = len(onset_frames) / (len(audio) / sr)
            

            rms = librosa.feature.rms(y=audio)[0]
            silence_threshold = np.percentile(rms, 20)
            pauses = rms < silence_threshold
            features['pause_ratio'] = np.sum(pauses) / len(pauses)
            

            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            
        except Exception as e:
            print(f"Error extracting prosodic features: {e}")
            features.update(self._get_default_features())
        
        return features
    
    def _extract_wav2vec_features(self, audio: np.ndarray) -> Dict:
        """Extract Wav2Vec2 embeddings"""
        try:
            inputs = self.wav2vec_processor(audio, sampling_rate=16000, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.wav2vec_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            

            features = {f'wav2vec_dim_{i}': embeddings[i] for i in range(min(50, len(embeddings)))}
            
        except Exception as e:
            print(f"Error extracting Wav2Vec2 features: {e}")
            features = {f'wav2vec_dim_{i}': 0.0 for i in range(50)}
        
        return features
    
    def _extract_emotion_features(self, audio_path: str) -> Dict:
        """Extract emotion predictions"""
        try:
            emotion_results = self.emotion_classifier(audio_path)
            features = {}
            
            for result in emotion_results:
                emotion_label = result['label'].lower().replace(' ', '_')
                features[f'emotion_{emotion_label}'] = result['score']
            
            return features
        except Exception as e:
            print(f"Error extracting emotion features: {e}")
            return {'emotion_neutral': 1.0}
    
    def _extract_voice_quality_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract voice quality features (jitter, shimmer approximations)"""
        features = {}
        
        try:

            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            

            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            

            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            

            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
                
        except Exception as e:
            print(f"Error extracting voice quality features: {e}")

            features.update({
                'spectral_centroid_mean': 0, 'spectral_centroid_std': 0,
                'spectral_rolloff_mean': 0, 'zcr_mean': 0, 'zcr_std': 0
            })
            for i in range(13):
                features[f'mfcc_{i}_mean'] = 0
                features[f'mfcc_{i}_std'] = 0
        
        return features


class EnhancedLinguisticAnalyzer:
    """Enhanced linguistic analysis with better filler word and disfluency detection"""
    
    def __init__(self):

        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.model = AutoModel.from_pretrained("bert-base-uncased")
            except:
                print("BERT model not available")
                self.tokenizer = None
                self.model = None
            

            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                                 model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            except:
                print("Sentiment analyzer not available")
                self.sentiment_analyzer = None
        else:
            self.tokenizer = None
            self.model = None
            self.sentiment_analyzer = None
        

        self.linguistic_categories = {
            'personal_pronouns': ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours'],
            'filler_words': [
                'um', 'uh', 'er', 'ah', 'like', 'you know', 'sort of', 'kind of', 
                'i mean', 'well', 'so', 'basically', 'actually', 'literally',
                'you see', 'right', 'okay', 'alright', 'anyway'
            ],
            'uncertainty_words': [
                'maybe', 'perhaps', 'possibly', 'might', 'could', 'probably',
                'i think', 'i guess', 'i believe', 'i suppose', 'seems like',
                'appears to be', 'looks like', 'sounds like', 'feels like',
                'should be', 'would be', 'may be', 'can be'
            ],
            'certainty_words': [
                'always', 'never', 'definitely', 'certainly', 'absolutely',
                'obviously', 'clearly', 'undoubtedly', 'surely', 'without doubt',
                'for sure', 'no doubt', 'of course', 'indeed'
            ],
            'negation_words': [
                'no', 'not', 'never', 'nothing', 'nobody', 'nowhere',
                'none', 'neither', 'nor', "don't", "won't", "can't",
                "shouldn't", "wouldn't", "couldn't", "isn't", "aren't"
            ],
            'cognitive_words': [
                'think', 'know', 'understand', 'realize', 'believe', 'remember',
                'forget', 'imagine', 'suppose', 'assume', 'consider', 'wonder'
            ],
            'hedging_words': [
                'somewhat', 'rather', 'quite', 'fairly', 'pretty', 'kinda',
                'sorta', 'more or less', 'to some extent', 'in a way'
            ],
            'intensifiers': [
                'very', 'really', 'extremely', 'totally', 'completely',
                'absolutely', 'incredibly', 'amazingly', 'seriously'
            ]
        }
    
    def extract_features(self, text: str, disfluency_features: Dict = None) -> Dict:
        """Extract comprehensive linguistic features including disfluencies"""
        print(f"DEBUG: Processing text: '{text}'")
        
        features = {}
        

        features.update(self._extract_basic_stats(text))
        

        features.update(self._extract_linguistic_categories(text))
        

        features.update(self._extract_disfluency_patterns(text))
        

        if disfluency_features:
            features.update(self._incorporate_audio_disfluencies(disfluency_features))
        

        if self.model is not None:
            features.update(self._extract_bert_features(text))
        

        if self.sentiment_analyzer is not None:
            features.update(self._extract_sentiment_features(text))
        

        features.update(self._extract_complexity_features(text))
        

        features.update(self._extract_deception_patterns(text))
        
        print("DEBUG: Final linguistic features:")
        for key, value in features.items():
            if 'ratio' in key and value > 0:
                print(f"  {key}: {value:.4f}")
        
        return features
    
    def _extract_basic_stats(self, text: str) -> Dict:
        """Extract basic text statistics"""
        words = text.lower().split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word.strip('.,!?;:')) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'unique_words_ratio': len(set(words)) / len(words) if words else 0
        }
        
        print(f"DEBUG: Basic stats - Words: {len(words)}, Sentences: {len(sentences)}")
        
        return features
    
    def _extract_linguistic_categories(self, text: str) -> Dict:
        """Extract enhanced linguistic categories"""

        text_clean = text.lower().strip()
        words = text_clean.split()
        total_words = len(words)
        

        text_phrases = text_clean
        
        print(f"DEBUG: Analyzing {total_words} words: {words}")
        
        features = {}
        
        for category, category_items in self.linguistic_categories.items():
            count = 0
            

            for word in words:
                cleaned_word = word.strip('.,!?;:"()[]')
                if cleaned_word in category_items:
                    count += 1
                    print(f"DEBUG: Found {category} word: '{cleaned_word}'")
            

            for phrase in category_items:
                if ' ' in phrase:  # Multi-word phrase
                    phrase_count = text_phrases.count(phrase)
                    count += phrase_count
                    if phrase_count > 0:
                        print(f"DEBUG: Found {category} phrase: '{phrase}' ({phrase_count} times)")
            
            ratio = count / total_words if total_words > 0 else 0
            features[f'{category}_ratio'] = ratio
            
            if count > 0:
                print(f"DEBUG: {category}: {count} occurrences, ratio: {ratio:.4f}")
        
        return features
    
    def _extract_disfluency_patterns(self, text: str) -> Dict:
        """Extract specific disfluency patterns"""
        features = {}
        

        words = text.lower().split()
        repetitions = 0
        
        for i in range(len(words) - 1):
            word1 = words[i].strip('.,!?;:"()')
            word2 = words[i + 1].strip('.,!?;:"()')
            if word1 == word2 and len(word1) > 2: 
                repetitions += 1
                print(f"DEBUG: Found repetition: '{word1}'")
        
        features['repetition_ratio'] = repetitions / len(words) if words else 0
        

        false_starts = text.count(',') + text.count(' - ') + text.count('...')
        features['false_start_ratio'] = false_starts / len(words) if words else 0
        

        corrections = (text.lower().count(' i mean ') + 
                      text.lower().count(' that is ') + 
                      text.lower().count(' or rather '))
        features['self_correction_ratio'] = corrections / len(words) if words else 0
        
        return features
    
    def _incorporate_audio_disfluencies(self, disfluency_features: Dict) -> Dict:
        """Incorporate audio-detected disfluencies into linguistic features"""
        features = {}
        

        estimated_duration = max(10, disfluency_features.get('total_disfluencies', 10))  # seconds
        
        features['audio_filled_pauses'] = disfluency_features.get('filled_pauses', 0) / estimated_duration
        features['audio_hesitation_pauses'] = disfluency_features.get('hesitation_pauses', 0) / estimated_duration
        features['audio_repetitions'] = disfluency_features.get('repeated_patterns', 0) / estimated_duration
        
        return features
    
    def _extract_bert_features(self, text: str) -> Dict:
        """Extract BERT embeddings"""
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            

            features = {f'bert_dim_{i}': embeddings[i] for i in range(min(50, len(embeddings)))}
            
        except Exception as e:
            print(f"Error extracting BERT features: {e}")
            features = {f'bert_dim_{i}': 0.0 for i in range(50)}
        
        return features
    
    def _extract_sentiment_features(self, text: str) -> Dict:
        """Extract sentiment analysis results"""
        try:
            sentiment_results = self.sentiment_analyzer(text)
            features = {}
            
            for result in sentiment_results:
                label = result['label'].lower()
                features[f'sentiment_{label}'] = result['score']
            
            return features
        except Exception as e:
            print(f"Error extracting sentiment features: {e}")
            return {'sentiment_neutral': 1.0}
    
    def _extract_complexity_features(self, text: str) -> Dict:
        """Extract text complexity features"""
        words = text.split()
        
        features = {
            'syllable_complexity': np.mean([self._count_syllables(word) for word in words]) if words else 0,
            'punctuation_ratio': sum(1 for char in text if char in '.,!?;:') / len(text) if text else 0,
        }
        
        return features
    
    def _extract_deception_patterns(self, text: str) -> Dict:
        """Extract patterns specifically associated with deception"""
        features = {}
        
        text_lower = text.lower()
        words = text_lower.split()
        

        detachment_words = ['that', 'there', 'those', 'them']
        detachment_count = sum(1 for word in words if word in detachment_words)
        features['detachment_ratio'] = detachment_count / len(words) if words else 0
        

        past_tense_markers = ['was', 'were', 'had', 'did', 'went', 'came', 'saw']
        past_tense_count = sum(1 for word in words if word in past_tense_markers)
        features['past_tense_ratio'] = past_tense_count / len(words) if words else 0
        

        specific_details = len(re.findall(r'\d+', text)) + text_lower.count('exactly') + text_lower.count('precisely')
        features['specific_details_ratio'] = specific_details / len(words) if words else 0
        
        return features
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting"""
        word = word.lower().strip('.,!?;:"()')
        vowels = 'aeiouy'
        syllable_count = 0
        previous_char_was_vowel = False
        
        for char in word:
            if char in vowels and not previous_char_was_vowel:
                syllable_count += 1
                previous_char_was_vowel = True
            else:
                previous_char_was_vowel = False
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)


class EnhancedDeceptionModel:
    """Enhanced deception detection model with better sensitivity"""
    
    def __init__(self):

        self.thresholds = {
            'blink_rate_high': 30,  
            'pitch_variation_high': 50,  
            'pause_ratio_high': 0.3,
            'filler_words_high': 0.05,
            'uncertainty_words_high': 0.03,  
            'repetition_high': 0.01,
            'audio_disfluencies_high': 0.1
        }
    
    def predict_deception(self, features: Dict) -> Dict:
        """Enhanced deception prediction with better feature integration"""
        

        facial_features = features.get('facial', {})
        audio_features = features.get('audio', {})
        text_features = features.get('linguistic', {})
        

        facial_score = self._calculate_facial_score(facial_features)
        audio_score = self._calculate_audio_score(audio_features)
        text_score = self._calculate_text_score(text_features)
        
        print(f"DEBUG: Individual scores - Facial: {facial_score:.3f}, Audio: {audio_score:.3f}, Text: {text_score:.3f}")
        

        if text_score > 0:

            combined_score = (facial_score * 0.4 + audio_score * 0.35 + text_score * 0.25)
        else:

            combined_score = (facial_score * 0.6 + audio_score * 0.4)
        

        enhanced_score = self._enhance_sensitivity(combined_score, facial_score, audio_score, text_score)
        
        print(f"DEBUG: Combined score: {combined_score:.3f}, Enhanced: {enhanced_score:.3f}")
        

        indicators = self._identify_indicators(facial_features, audio_features, text_features)
        

        detection_threshold = self._calculate_dynamic_threshold(indicators, enhanced_score)
        
        return {
            'deception_probability': enhanced_score,
            'deception_detected': enhanced_score > detection_threshold,
            'confidence': min(0.95, abs(enhanced_score - 0.5) * 2),
            'detection_threshold': detection_threshold,
            'modality_scores': {
                'facial': facial_score,
                'audio': audio_score,
                'linguistic': text_score
            },
            'key_indicators': indicators
        }
    
    def _enhance_sensitivity(self, base_score: float, facial_score: float, audio_score: float, text_score: float) -> float:
        """Apply sensitivity enhancement based on strong individual indicators"""
        

        max_individual = max(facial_score, audio_score, text_score)
        
        if max_individual > 0.7:

            boost = (max_individual - 0.7) * 0.5
            enhanced = min(1.0, base_score + boost)
        elif max_individual > 0.5 and base_score > 0.3:

            boost = (max_individual - 0.5) * 0.2
            enhanced = min(1.0, base_score + boost)
        else:
            enhanced = base_score
        
        return enhanced
    
    def _calculate_dynamic_threshold(self, indicators: List[str], score: float) -> float:
        """Calculate dynamic threshold based on indicator strength"""
        

        base_threshold = 0.5
        

        strong_indicators = [ind for ind in indicators if any(keyword in ind.lower() 
                           for keyword in ['high', 'frequent', 'excessive', 'elevated'])]
        
        if len(strong_indicators) >= 3:
            return 0.35  # Lower threshold for multiple indicators
        elif len(strong_indicators) >= 2:
            return 0.4   # Slightly lower threshold
        elif len(indicators) >= 2:
            return 0.45  # Any two indicators
        else:
            return base_threshold
    
    def _calculate_facial_score(self, features: Dict) -> float:
        """Enhanced facial deception score calculation"""
        blink_rate = features.get('blink_rate', 0)
        micro_expressions = features.get('micro_expressions', [0])
        

        if blink_rate > 30:
            blink_score = 1.0  # Very high
        elif blink_rate > self.thresholds['blink_rate_high']:
            blink_score = 0.8  # High
        elif blink_rate > 15:
            blink_score = 0.6  # Moderate
        else:
            blink_score = max(0, (blink_rate - 10) / 20)  # Scale from normal
        

        if isinstance(micro_expressions, list) and len(micro_expressions) > 1:
            micro_variance = np.var(micro_expressions)
            micro_score = min(1.0, micro_variance * 2000)  # Increased sensitivity
        else:
            micro_score = 0
        
        return min(1.0, (blink_score * 0.7 + micro_score * 0.3))
    
    def _calculate_audio_score(self, features: Dict) -> float:
        """Enhanced audio deception score calculation"""
        pitch_std = features.get('pitch_std', 0)
        pause_ratio = features.get('pause_ratio', 0)
        speech_rate = features.get('speech_rate', 0)
        

        pitch_score = min(1.0, pitch_std / self.thresholds['pitch_variation_high'])
        

        if pause_ratio > self.thresholds['pause_ratio_high']:
            pause_score = 1.0
        else:
            pause_score = pause_ratio / self.thresholds['pause_ratio_high']
        

        if speech_rate > 0:
            normal_rate = 2.5  # words per second
            rate_deviation = abs(speech_rate - normal_rate) / normal_rate
            rate_score = min(1.0, rate_deviation)
        else:
            rate_score = 0
        
        return min(1.0, (pitch_score * 0.4 + pause_score * 0.4 + rate_score * 0.2))
    
    def _calculate_text_score(self, features: Dict) -> float:
        """Enhanced text deception score calculation"""
        filler_ratio = features.get('filler_words_ratio', 0)
        uncertainty_ratio = features.get('uncertainty_words_ratio', 0)
        negation_ratio = features.get('negation_words_ratio', 0)
        repetition_ratio = features.get('repetition_ratio', 0)
        

        filler_score = min(1.0, filler_ratio / self.thresholds['filler_words_high'])
        uncertainty_score = min(1.0, uncertainty_ratio / self.thresholds['uncertainty_words_high'])
        negation_score = min(1.0, negation_ratio * 30)  # Scale up
        repetition_score = min(1.0, repetition_ratio / self.thresholds['repetition_high'])
        

        audio_fillers = features.get('audio_filled_pauses', 0)
        audio_score = min(1.0, audio_fillers / self.thresholds['audio_disfluencies_high'])
        
        return min(1.0, (filler_score * 0.25 + uncertainty_score * 0.25 + 
                        negation_score * 0.15 + repetition_score * 0.15 + 
                        audio_score * 0.2))
    
    def _identify_indicators(self, facial_features: Dict, audio_features: Dict, text_features: Dict) -> List[str]:
        """Identify specific deception indicators"""
        indicators = []
        

        blink_rate = facial_features.get('blink_rate', 0)
        if blink_rate > self.thresholds['blink_rate_high']:
            indicators.append(f"Elevated blink rate: {blink_rate:.1f}/min")
        

        pitch_std = audio_features.get('pitch_std', 0)
        if pitch_std > self.thresholds['pitch_variation_high']:
            indicators.append("High vocal pitch variability")
        
        pause_ratio = audio_features.get('pause_ratio', 0)
        if pause_ratio > self.thresholds['pause_ratio_high']:
            indicators.append(f"Frequent speech pauses: {pause_ratio:.1%}")
        

        filler_ratio = text_features.get('filler_words_ratio', 0)
        if filler_ratio > self.thresholds['filler_words_high']:
            indicators.append(f"High filler word usage: {filler_ratio:.1%}")
        
        uncertainty_ratio = text_features.get('uncertainty_words_ratio', 0)
        if uncertainty_ratio > self.thresholds['uncertainty_words_high']:
            indicators.append(f"Frequent uncertainty language: {uncertainty_ratio:.1%}")
        
        repetition_ratio = text_features.get('repetition_ratio', 0)
        if repetition_ratio > self.thresholds['repetition_high']:
            indicators.append(f"Speech repetitions detected: {repetition_ratio:.1%}")
        

        audio_fillers = text_features.get('audio_filled_pauses', 0)
        if audio_fillers > self.thresholds['audio_disfluencies_high']:
            indicators.append("Audio-detected speech disfluencies")
        
        return indicators


class EnhancedDeceptionDetector:
    """Enhanced main deception detection system"""
    
    def __init__(self):
        self.facial_analyzer = FacialAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.linguistic_analyzer = EnhancedLinguisticAnalyzer()
        self.speech_converter = EnhancedSpeechToTextConverter()
        self.model = EnhancedDeceptionModel()
        
    def extract_all_features(self, video_path: str, audio_path: str, text: str, 
                           disfluency_features: Dict = None) -> Dict:
        """Extract features from all modalities with enhanced integration"""
        print("Extracting facial features...")
        facial_features = self.facial_analyzer.extract_features(video_path)
        
        print("Extracting audio features...")
        audio_features = self.audio_analyzer.extract_features(audio_path) if audio_path else {}
        
        print("Extracting linguistic features...")
        text_features = self.linguistic_analyzer.extract_features(text, disfluency_features)
        
        all_features = {
            'facial': facial_features,
            'audio': audio_features,
            'linguistic': text_features,
        }
        
        return all_features
    
    def detect_deception(self, video_path: str, audio_path: str, text: str = None) -> Dict:
        """Enhanced deception detection pipeline"""
        

        if not text and audio_path:
            print("Performing enhanced transcription...")
            text, disfluency_features = self.speech_converter.convert_audio_to_text_with_fillers(audio_path)
            print(f"Enhanced transcription result: {text}")
        else:
            disfluency_features = {}
        

        all_features = self.extract_all_features(video_path, audio_path, text, disfluency_features)
        

        result = self.model.predict_deception(all_features)
        

        result['detailed_features'] = all_features
        result['transcribed_text'] = text
        result['disfluency_features'] = disfluency_features
        
        return result


class EnhancedDeceptionDetectionApp:
    """Enhanced GUI Application with better transcription and analysis"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced Multimodal Deception Detection System")
        self.root.geometry("900x800")
        
        self.camera_recorder = CameraRecorder()
        self.detector = EnhancedDeceptionDetector()
        
        self.is_recording = False
        self.current_video_path = None
        self.current_audio_path = None
        self.current_text = ""
        self.disfluency_features = {}
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the enhanced GUI elements"""

        title_label = tk.Label(self.root, text="Enhanced Multimodal Deception Detection System", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        

        status_frame = ttk.Frame(self.root)
        status_frame.pack(pady=5, padx=20, fill='x')
        

        available_features = []
        if MEDIAPIPE_AVAILABLE:
            available_features.append("Facial Analysis")
        if LIBROSA_AVAILABLE:
            available_features.append("Audio Analysis")
        if TRANSFORMERS_AVAILABLE:
            available_features.append("AI Models")
        if PYAUDIO_AVAILABLE:
            available_features.append("Audio Recording")
        if SPEECH_RECOGNITION_AVAILABLE:
            available_features.append("Speech-to-Text")
        if WHISPER_DIRECT_AVAILABLE:
            available_features.append("Enhanced Whisper")
        
        total_possible = 6
        if len(available_features) == total_possible:
            status_label = tk.Label(status_frame,
                                   text="✓ All dependencies loaded - Full enhanced functionality available",
                                   font=("Arial", 10), foreground="green")
        elif len(available_features) >= 4:
            status_text = f"✓ Enhanced functionality: {len(available_features)}/{total_possible} features available"
            status_label = tk.Label(status_frame, text=status_text,
                                   font=("Arial", 10), foreground="blue")
        else:
            status_text = f"⚠ Limited functionality: {len(available_features)}/{total_possible} features available"
            status_label = tk.Label(status_frame, text=status_text,
                                   font=("Arial", 10), foreground="orange")
        status_label.pack()
        

        if WHISPER_DIRECT_AVAILABLE:
            whisper_label = tk.Label(status_frame, 
                                   text="✓ Direct Whisper available - Better filler word preservation",
                                   font=("Arial", 8), foreground="green")
            whisper_label.pack()
        else:
            whisper_label = tk.Label(status_frame,
                                   text="⚠ Install 'openai-whisper' for better filler word detection",
                                   font=("Arial", 8), foreground="orange")
            whisper_label.pack()
        

        recording_frame = ttk.Frame(self.root)
        recording_frame.pack(pady=10, padx=20, fill='x')
        
        ttk.Label(recording_frame, text="Step 1: Record Video", 
                 font=("Arial", 12, "bold")).pack(anchor='w')
        
        self.record_button = ttk.Button(recording_frame, text="Start Recording", 
                                       command=self.toggle_recording)
        self.record_button.pack(pady=5)
        
        if not (cv2.__version__ and PYAUDIO_AVAILABLE):
            self.record_button.config(state='disabled')
            ttk.Label(recording_frame, text="Recording disabled - missing opencv or pyaudio",
                     foreground="red").pack()
        
        self.recording_status = ttk.Label(recording_frame, text="Ready to record", 
                                         foreground="blue")
        self.recording_status.pack()
        

        file_frame = ttk.Frame(self.root)
        file_frame.pack(pady=10, padx=20, fill='x')
        
        ttk.Label(file_frame, text="Or load existing files:", 
                 font=("Arial", 10)).pack(anchor='w')
        
        file_buttons_frame = ttk.Frame(file_frame)
        file_buttons_frame.pack(fill='x')
        
        ttk.Button(file_buttons_frame, text="Load Video", 
                  command=self.load_video_file).pack(side='left', padx=5)
        ttk.Button(file_buttons_frame, text="Load Audio", 
                  command=self.load_audio_file).pack(side='left', padx=5)
        

        text_frame = ttk.Frame(self.root)
        text_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        ttk.Label(text_frame, text="Step 2: Enhanced Transcribed Text", 
                 font=("Arial", 12, "bold")).pack(anchor='w')
        

        info_label = ttk.Label(text_frame, 
                              text="Enhanced transcription preserves 'um', 'uh', 'like', and other speech patterns",
                              font=("Arial", 8), foreground="blue")
        info_label.pack(anchor='w')
        
        self.text_area = tk.Text(text_frame, height=8, wrap='word')
        self.text_area.pack(fill='both', expand=True, pady=5)
        
        text_buttons_frame = ttk.Frame(text_frame)
        text_buttons_frame.pack(fill='x')
        
        transcribe_button = ttk.Button(text_buttons_frame, text="Enhanced Auto-Transcribe", 
                                      command=self.enhanced_transcribe_audio)
        if not (SPEECH_RECOGNITION_AVAILABLE or WHISPER_DIRECT_AVAILABLE):
            transcribe_button.config(state='disabled')
        transcribe_button.pack(side='left', padx=5)
        
        ttk.Button(text_buttons_frame, text="Clear Text", 
                  command=lambda: self.text_area.delete('1.0', tk.END)).pack(side='left', padx=5)
        

        disfluency_frame = ttk.Frame(self.root)
        disfluency_frame.pack(pady=5, padx=20, fill='x')
        
        self.disfluency_info = ttk.Label(disfluency_frame, text="", 
                                        font=("Arial", 8), foreground="gray")
        self.disfluency_info.pack()
        

        analysis_frame = ttk.Frame(self.root)
        analysis_frame.pack(pady=10, padx=20, fill='x')
        
        ttk.Label(analysis_frame, text="Step 3: Enhanced Analysis", 
                 font=("Arial", 12, "bold")).pack(anchor='w')
        
        analysis_info = ttk.Label(analysis_frame,
                                 text="Enhanced model with improved sensitivity and dynamic thresholds",
                                 font=("Arial", 8), foreground="blue")
        analysis_info.pack(anchor='w')
        
        self.analyze_button = ttk.Button(analysis_frame, text="Run Enhanced Analysis", 
                                        command=self.run_enhanced_analysis)
        self.analyze_button.pack(pady=5)
        
        self.analysis_status = ttk.Label(analysis_frame, text="Ready for enhanced analysis")
        self.analysis_status.pack()
        

        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(pady=5, padx=20, fill='x')
        
    def toggle_recording(self):
        """Toggle recording state"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start video recording"""
        if self.camera_recorder.start_recording():
            self.is_recording = True
            self.record_button.config(text="Stop Recording")
            self.recording_status.config(text="Recording... Press 'Q' in video window to stop", 
                                       foreground="red")
            self.analysis_status.config(text="")
            self.disfluency_info.config(text="")
        else:
            messagebox.showerror("Error", "Failed to start recording. Check camera connection.")
    
    def stop_recording(self):
        """Stop video recording"""
        video_path, audio_path = self.camera_recorder.stop_recording()
        
        self.is_recording = False
        self.record_button.config(text="Start Recording")
        self.recording_status.config(text="Recording completed", foreground="green")
        
        if video_path:
            self.current_video_path = video_path
            self.current_audio_path = audio_path
            

            if audio_path:
                self.enhanced_transcribe_audio()
        else:
            messagebox.showerror("Error", "Failed to save recording files")
    
    def load_video_file(self):
        """Load video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.current_video_path = file_path
            self.recording_status.config(text=f"Video loaded: {os.path.basename(file_path)}")
    
    def load_audio_file(self):
        """Load audio file"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.m4a *.flac"), ("All files", "*.*")]
        )
        if file_path:
            self.current_audio_path = file_path
            self.recording_status.config(text=f"Audio loaded: {os.path.basename(file_path)}")
    
    def enhanced_transcribe_audio(self):
        """Enhanced transcription with filler word preservation"""
        if not self.current_audio_path:
            messagebox.showwarning("Warning", "No audio file available for transcription")
            return
            
        if not (SPEECH_RECOGNITION_AVAILABLE or WHISPER_DIRECT_AVAILABLE):
            messagebox.showwarning("Warning", "Enhanced speech recognition not available")
            return
        
        self.analysis_status.config(text="Running enhanced transcription...")
        self.progress.start()
        
        def transcribe():
            try:

                text, disfluency_features = self.detector.speech_converter.convert_audio_to_text_with_fillers(
                    self.current_audio_path
                )
                self.root.after(0, self.update_enhanced_transcription, text, disfluency_features)
            except Exception as e:
                self.root.after(0, self.transcription_error, str(e))
        
        thread = threading.Thread(target=transcribe)
        thread.start()
    
    def update_enhanced_transcription(self, text, disfluency_features):
        """Update text area with enhanced transcription"""
        self.progress.stop()
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', text)
        self.current_text = text
        self.disfluency_features = disfluency_features
        

        total_disfluencies = disfluency_features.get('total_disfluencies', 0)
        filled_pauses = disfluency_features.get('filled_pauses', 0)
        
        disfluency_text = f"Detected: {total_disfluencies} total disfluencies, {filled_pauses} filled pauses"
        self.disfluency_info.config(text=disfluency_text)
        
        self.analysis_status.config(text="Enhanced transcription completed")
    
    def transcription_error(self, error_msg):
        """Handle transcription error"""
        self.progress.stop()
        self.analysis_status.config(text="Enhanced transcription failed")
        messagebox.showerror("Transcription Error", f"Failed to transcribe audio: {error_msg}")
    
    def run_enhanced_analysis(self):
        """Run enhanced deception detection analysis"""

        self.current_text = self.text_area.get('1.0', tk.END).strip()
        

        if not self.current_video_path:
            messagebox.showwarning("Warning", "Please record or load a video file")
            return
        
        if not self.current_text:
            messagebox.showwarning("Warning", "Please provide text for analysis")
            return
        
        self.analysis_status.config(text="Running enhanced analysis... This may take a moment")
        self.progress.start()
        self.analyze_button.config(state='disabled')
        
        def analyze():
            try:
                result = self.detector.detect_deception(
                    self.current_video_path, 
                    self.current_audio_path, 
                    self.current_text
                )
                self.root.after(0, self.show_enhanced_results, result)
            except Exception as e:
                self.root.after(0, self.analysis_error, str(e))
        
        thread = threading.Thread(target=analyze)
        thread.start()
    
    def show_enhanced_results(self, result):
        """Display enhanced analysis results"""
        self.progress.stop()
        self.analyze_button.config(state='disabled')
        self.analysis_status.config(text="Enhanced analysis completed")
        

        results_window = tk.Toplevel(self.root)
        results_window.title("Enhanced Deception Detection Results")
        results_window.geometry("700x600")
        

        main_frame = ttk.Frame(results_window)
        main_frame.pack(pady=10, padx=20, fill='both', expand=True)
        

        prob_text = f"Deception Probability: {result['deception_probability']:.1%}"
        prob_color = "red" if result['deception_detected'] else "green"
        
        prob_label = tk.Label(main_frame, text=prob_text, font=("Arial", 16, "bold"), 
                             foreground=prob_color)
        prob_label.pack()
        

        detection_text = "DECEPTION DETECTED" if result['deception_detected'] else "NO DECEPTION DETECTED"
        detection_label = tk.Label(main_frame, text=detection_text, font=("Arial", 14), 
                                  foreground=prob_color)
        detection_label.pack(pady=5)
        

        confidence_text = f"Confidence: {result['confidence']:.1%}"
        threshold_text = f"Dynamic Threshold: {result['detection_threshold']:.1%}"
        
        confidence_label = tk.Label(main_frame, text=confidence_text)
        confidence_label.pack()
        
        threshold_label = tk.Label(main_frame, text=threshold_text, font=("Arial", 8), 
                                  foreground="gray")
        threshold_label.pack()
        

        scores_label = tk.Label(main_frame, text="Enhanced Modality Scores:", 
                               font=("Arial", 12, "bold"))
        scores_label.pack(pady=(15,5))
        
        for modality, score in result['modality_scores'].items():
            score_text = f"{modality.capitalize()}: {score:.1%}"
            color = "red" if score > 0.6 else "orange" if score > 0.4 else "green"
            score_label = tk.Label(main_frame, text=score_text, foreground=color)
            score_label.pack()
        

        if result['key_indicators']:
            indicators_title = tk.Label(main_frame, text="Detected Indicators:", 
                                       font=("Arial", 12, "bold"))
            indicators_title.pack(pady=(15,5))
            

            indicators_frame = ttk.Frame(main_frame)
            indicators_frame.pack(fill='both', expand=True, pady=5)
            
            indicators_text = tk.Text(indicators_frame, height=6, wrap='word')
            indicators_scrollbar = ttk.Scrollbar(indicators_frame, orient='vertical', 
                                               command=indicators_text.yview)
            indicators_text.configure(yscrollcommand=indicators_scrollbar.set)
            
            indicators_text.pack(side='left', fill='both', expand=True)
            indicators_scrollbar.pack(side='right', fill='y')
            
            for indicator in result['key_indicators']:
                indicators_text.insert(tk.END, f"• {indicator}\n")
            
            indicators_text.config(state='disabled')
        

        button_frame = ttk.Frame(results_window)
        button_frame.pack(pady=10)
        
        save_button = ttk.Button(button_frame, text="Save Enhanced Report", 
                                command=lambda: self.save_enhanced_report(result))
        save_button.pack(side='left', padx=5)
        
        details_button = ttk.Button(button_frame, text="View Details", 
                                   command=lambda: self.show_detailed_features(result))
        details_button.pack(side='left', padx=5)
        
        close_button = ttk.Button(button_frame, text="Close", 
                                 command=results_window.destroy)
        close_button.pack(side='left', padx=5)
        

        self.analyze_button.config(state='normal')
    
    def show_detailed_features(self, result):
        """Show detailed feature breakdown"""
        details_window = tk.Toplevel(self.root)
        details_window.title("Detailed Feature Analysis")
        details_window.geometry("800x600")
        

        notebook = ttk.Notebook(details_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        

        facial_frame = ttk.Frame(notebook)
        notebook.add(facial_frame, text="Facial Analysis")
        
        facial_text = tk.Text(facial_frame, wrap='word')
        facial_scroll = ttk.Scrollbar(facial_frame, orient='vertical', command=facial_text.yview)
        facial_text.configure(yscrollcommand=facial_scroll.set)
        
        facial_text.pack(side='left', fill='both', expand=True)
        facial_scroll.pack(side='right', fill='y')
        
        facial_features = result['detailed_features']['facial']
        facial_text.insert(tk.END, "FACIAL ANALYSIS FEATURES:\n\n")
        for key, value in facial_features.items():
            if isinstance(value, (int, float)):
                facial_text.insert(tk.END, f"{key}: {value:.3f}\n")
            elif isinstance(value, list) and len(value) < 10:
                facial_text.insert(tk.END, f"{key}: {value}\n")
        

        audio_frame = ttk.Frame(notebook)
        notebook.add(audio_frame, text="Audio Analysis")
        
        audio_text = tk.Text(audio_frame, wrap='word')
        audio_scroll = ttk.Scrollbar(audio_frame, orient='vertical', command=audio_text.yview)
        audio_text.configure(yscrollcommand=audio_scroll.set)
        
        audio_text.pack(side='left', fill='both', expand=True)
        audio_scroll.pack(side='right', fill='y')
        
        audio_features = result['detailed_features']['audio']
        audio_text.insert(tk.END, "AUDIO ANALYSIS FEATURES:\n\n")
        for key, value in audio_features.items():
            if isinstance(value, (int, float)):
                audio_text.insert(tk.END, f"{key}: {value:.3f}\n")
        

        linguistic_frame = ttk.Frame(notebook)
        notebook.add(linguistic_frame, text="Linguistic Analysis")
        
        linguistic_text = tk.Text(linguistic_frame, wrap='word')
        linguistic_scroll = ttk.Scrollbar(linguistic_frame, orient='vertical', 
                                         command=linguistic_text.yview)
        linguistic_text.configure(yscrollcommand=linguistic_scroll.set)
        
        linguistic_text.pack(side='left', fill='both', expand=True)
        linguistic_scroll.pack(side='right', fill='y')
        
        linguistic_features = result['detailed_features']['linguistic']
        linguistic_text.insert(tk.END, "LINGUISTIC ANALYSIS FEATURES:\n\n")
        linguistic_text.insert(tk.END, f"Transcribed Text: {result['transcribed_text']}\n\n")
        
        for key, value in linguistic_features.items():
            if isinstance(value, (int, float)):
                linguistic_text.insert(tk.END, f"{key}: {value:.4f}\n")
        

        facial_text.config(state='disabled')
        audio_text.config(state='disabled')
        linguistic_text.config(state='disabled')
    
    def save_enhanced_report(self, result):
        """Save enhanced analysis report"""
        file_path = filedialog.asksaveasfilename(
            title="Save Enhanced Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("ENHANCED DECEPTION DETECTION REPORT\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Deception Probability: {result['deception_probability']:.1%}\n")
                    f.write(f"Deception Detected: {result['deception_detected']}\n")
                    f.write(f"Confidence: {result['confidence']:.1%}\n")
                    f.write(f"Dynamic Threshold Used: {result['detection_threshold']:.1%}\n\n")
                    
                    f.write("ENHANCED MODALITY SCORES:\n")
                    for modality, score in result['modality_scores'].items():
                        f.write(f"  {modality.capitalize()}: {score:.1%}\n")
                    
                    if result['key_indicators']:
                        f.write("\nDETECTED INDICATORS:\n")
                        for indicator in result['key_indicators']:
                            f.write(f"  • {indicator}\n")
                    

                    if result.get('disfluency_features'):
                        f.write("\nDISFLUENCY ANALYSIS:\n")
                        for key, value in result['disfluency_features'].items():
                            f.write(f"  {key}: {value}\n")
                    
                    f.write(f"\nTRANSCRIBED TEXT:\n{result['transcribed_text']}\n")
                    

                    f.write("\nDETAILED FEATURE SUMMARY:\n")
                    f.write("-" * 30 + "\n")
                    
                    for modality, features in result['detailed_features'].items():
                        f.write(f"\n{modality.upper()} FEATURES:\n")
                        for key, value in features.items():
                            if isinstance(value, (int, float)) and not key.startswith('bert_') and not key.startswith('wav2vec_'):
                                f.write(f"  {key}: {value:.4f}\n")
                
                messagebox.showinfo("Success", f"Enhanced report saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report: {e}")
    
    def analysis_error(self, error_msg):
        """Handle analysis error"""
        self.progress.stop()
        self.analyze_button.config(state='normal')
        self.analysis_status.config(text="Enhanced analysis failed")
        messagebox.showerror("Analysis Error", f"Failed to run enhanced analysis: {error_msg}")
    
    def run(self):
        """Start the enhanced application"""
        self.root.mainloop()



if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED MULTIMODAL DECEPTION DETECTION SYSTEM")
    print("=" * 70)
    print("Checking dependencies...")
    print()
    

    dependencies = {
        "OpenCV (Video)": cv2.__version__ if 'cv2' in globals() else "NOT FOUND",
        "MediaPipe (Facial)": "Available" if MEDIAPIPE_AVAILABLE else "Missing",
        "Librosa (Audio)": "Available" if LIBROSA_AVAILABLE else "Missing", 
        "PyAudio (Recording)": "Available" if PYAUDIO_AVAILABLE else "Missing",
        "SpeechRecognition": "Available" if SPEECH_RECOGNITION_AVAILABLE else "Missing",
        "Transformers (AI)": "Available" if TRANSFORMERS_AVAILABLE else "Missing",
        "Scikit-learn": "Available" if SKLEARN_AVAILABLE else "Missing",
        "OpenAI Whisper (Direct)": "Available" if WHISPER_DIRECT_AVAILABLE else "Missing - RECOMMENDED",
    }
    
    for name, status in dependencies.items():
        if "Missing" in status or status == "NOT FOUND":
            if "RECOMMENDED" in status:
                print(f"⚠️  {name}: {status}")
            else:
                print(f"❌ {name}: {status}")
        else:
            print(f"✅ {name}: {status}")
    
    print()
    

    missing_packages = []
    if not MEDIAPIPE_AVAILABLE:
        missing_packages.append("mediapipe")
    if not LIBROSA_AVAILABLE:
        missing_packages.append("librosa") 
    if not PYAUDIO_AVAILABLE:
        missing_packages.append("pyaudio")
    if not SPEECH_RECOGNITION_AVAILABLE:
        missing_packages.append("speechrecognition")
    if not SKLEARN_AVAILABLE:
        missing_packages.extend(["scikit-learn", "pandas"])
    
    if missing_packages:
        print("To install missing packages, run:")
        print(f"pip install {' '.join(set(missing_packages))}")
        print()
    
    if not TRANSFORMERS_AVAILABLE:
        print("To fix Transformers (protobuf issue):")
        print("pip uninstall protobuf tensorflow transformers")
        print("pip install protobuf==3.20.3 tensorflow==2.13.0 transformers")
        print()
    
    if not WHISPER_DIRECT_AVAILABLE:
        print("📢 RECOMMENDED: For best filler word detection, install:")
        print("pip install openai-whisper")
        print("This significantly improves transcription of 'um', 'uh', 'like', etc.")
        print()
    
    print("🚀 Starting Enhanced Application...")
    print()
    print("ENHANCEMENTS:")
    print("- Better filler word preservation ('um', 'uh', 'like')")
    print("- Audio-based disfluency detection")  
    print("- Dynamic detection thresholds")
    print("- Improved sensitivity to deception indicators")
    print("- Enhanced linguistic analysis")
    print()
    
    try:
        app = EnhancedDeceptionDetectionApp()
        app.run()
    except Exception as e:
        print(f"Error starting enhanced application: {e}")
        print("Please install missing dependencies and try again.")
