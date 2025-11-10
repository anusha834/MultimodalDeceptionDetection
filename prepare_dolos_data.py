import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
import sys

# Import your feature extractors
sys.path.append('..')  # Adjust to your project structure
from your_detection_system import (
    FacialAnalyzer, 
    AudioAnalyzer, 
    EnhancedLinguisticAnalyzer,
    EnhancedSpeechToTextConverter
)

class DOLOSDataProcessor:
    """Process DOLOS dataset into features for ML training"""
    
    def __init__(self, dolos_excel_path: str, videos_folder: str):
        self.df = pd.read_excel(dolos_excel_path)
        self.videos_folder = videos_folder
        
        # Initialize extractors
        self.facial_analyzer = FacialAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.linguistic_analyzer = EnhancedLinguisticAnalyzer()
        self.speech_converter = EnhancedSpeechToTextConverter()
    
    def extract_features_from_video(self, video_path: str, audio_path: str = None):
        """Extract all features from a single video"""
        
        print(f"Processing: {video_path}")
        
        # Extract facial features
        facial_features = self.facial_analyzer.extract_features(video_path)
        
        # Extract audio features
        if audio_path and os.path.exists(audio_path):
            audio_features = self.audio_analyzer.extract_features(audio_path)
            
            # Transcribe and get linguistic features
            text, disfluency = self.speech_converter.convert_audio_to_text_with_fillers(audio_path)
            linguistic_features = self.linguistic_analyzer.extract_features(text, disfluency)
        else:
            audio_features = {}
            linguistic_features = {}
        
        return {
            'facial': facial_features,
            'audio': audio_features,
            'linguistic': linguistic_features
        }
    
    def process_dataset(self, output_file: str = 'dolos_features.npz'):
        """Process entire DOLOS dataset"""
        
        X_list = []
        y_list = []
        metadata_list = []
        
        for idx, row in self.df.iterrows():
            try:
                video_file = os.path.join(self.videos_folder, row['video_filename'])
                audio_file = video_file.replace('.mp4', '.wav')
                
                if not os.path.exists(video_file):
                    print(f"Skipping {video_file} - not found")
                    continue
                
                # Extract features
                features = self.extract_features_from_video(video_file, audio_file)
                
                # Convert to feature vector (import from trainable_model.py)
                from trainable_model import TrainableDeceptionModel
                model = TrainableDeceptionModel()
                feature_vector = model.prepare_features(features)
                
                X_list.append(feature_vector)
                y_list.append(1 if row['label'] == 'deceptive' else 0)
                
                metadata_list.append({
                    'video_id': row.get('video_id', idx),
                    'subject_id': row.get('subject_id', ''),
                    'gender': row.get('gender', ''),
                })
                
                print(f"✓ Processed {idx+1}/{len(self.df)}")
                
            except Exception as e:
                print(f"✗ Error processing {idx}: {e}")
                continue
        
        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Save processed data
        np.savez(output_file, 
                 X=X, 
                 y=y, 
                 metadata=metadata_list,
                 feature_names=model.feature_names)
        
        print(f"\n✓ Dataset processed and saved to {output_file}")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Truthful: {np.sum(y==0)}, Deceptive: {np.sum(y==1)}")
        
        return X, y

# Usage
if __name__ == "__main__":
    processor = DOLOSDataProcessor(
        dolos_excel_path='Dolos.xlsx',
        videos_folder='dolos_videos/'
    )
    
    X, y = processor.process_dataset('dolos_features.npz')
