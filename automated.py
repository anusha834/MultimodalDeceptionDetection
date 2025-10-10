"""
Automated Data Preparation for Deception Detection Training

This script helps you:
1. Download sample data or prepare your own
2. Extract features from videos
3. Create training-ready datasets
4. Train models automatically

Usage:
    python prepare_training_data.py --mode [download|extract|train]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DatasetPreparer:
    """Automated dataset preparation for deception detection"""
    
    def __init__(self, output_dir: str = 'training_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.features_dir = self.output_dir / 'features'
        self.models_dir = self.output_dir / 'models'
        self.features_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
    
    def create_sample_dataset_info(self):
        """Create a guide for obtaining datasets"""
        
        datasets = {
            "DOLOS": {
                "url": "https://rose1.ntu.edu.sg/dataset/DOLOS/",
                "description": "1,675 video clips from reality TV shows",
                "size": "~50GB",
                "access": "Free for academic research (request form required)",
                "recommended": True
            },
            "MDPE": {
                "url": "https://arxiv.org/abs/2407.12274",
                "description": "Multimodal with personality/emotion annotations",
                "size": "TBD",
                "access": "Check ArXiv for availability",
                "recommended": True
            },
            "Box of Lies": {
                "url": "Search: Jimmy Fallon Box of Lies dataset",
                "description": "Celebrity interview clips",
                "size": "Small (~100 clips)",
                "access": "Public (extract from YouTube)",
                "recommended": False
            }
        }
        
        print("\n" + "="*70)
        print("AVAILABLE DECEPTION DETECTION DATASETS")
        print("="*70)
        
        for name, info in datasets.items():
            print(f"\n{name}:")
            print(f"  URL: {info['url']}")
            print(f"  Description: {info['description']}")
            print(f"  Size: {info['size']}")
            print(f"  Access: {info['access']}")
            print(f"  Recommended: {'✓ YES' if info['recommended'] else '  No'}")
        
        # Save to file
        info_file = self.output_dir / 'dataset_sources.json'
        with open(info_file, 'w') as f:
            json.dump(datasets, f, indent=2)
        
        print(f"\n✓ Dataset info saved to: {info_file}")
        
        return datasets
    
    def create_directory_structure(self):
        """Create organized directory structure for data"""
        
        dirs = [
            'raw_videos',
            'audio_extracted',
            'features',
            'models',
            'logs'
        ]
        
        for dir_name in dirs:
            dir_path = self.output_dir / dir_name
            dir_path.mkdir(exist_ok=True)
        
        print("\n✓ Created directory structure:")
        for dir_name in dirs:
            print(f"  - {self.output_dir / dir_name}")
        
        # Create README
        readme = self.output_dir / 'README.txt'
        with open(readme, 'w') as f:
            f.write("DECEPTION DETECTION TRAINING DATA\n")
            f.write("="*50 + "\n\n")
            f.write("Directory Structure:\n")
            f.write("  raw_videos/     - Place your video files here\n")
            f.write("  audio_extracted/ - Extracted audio files (auto-generated)\n")
            f.write("  features/       - Extracted features (auto-generated)\n")
            f.write("  models/         - Trained models (auto-generated)\n")
            f.write("  logs/           - Processing logs\n\n")
            f.write("Instructions:\n")
            f.write("1. Place videos in raw_videos/\n")
            f.write("2. Create labels.csv with columns: filename, label (0=truth, 1=deception)\n")
            f.write("3. Run: python prepare_training_data.py --mode extract\n")
            f.write("4. Run: python prepare_training_data.py --mode train\n")
        
        print(f"\n✓ Created README: {readme}")
    
    def extract_audio_from_videos(self):
        """Extract audio tracks from video files"""
        try:
            import moviepy.editor as mp
        except ImportError:
            print("❌ MoviePy not installed. Install with: pip install moviepy")
            return False
        
        video_dir = self.output_dir / 'raw_videos'
        audio_dir = self.output_dir / 'audio_extracted'
        
        video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
        
        if not video_files:
            print(f"❌ No video files found in {video_dir}")
            return False
        
        print(f"\nExtracting audio from {len(video_files)} videos...")
        
        for video_file in tqdm(video_files, desc="Extracting audio"):
            try:
                audio_file = audio_dir / f"{video_file.stem}.wav"
                
                if audio_file.exists():
                    continue
                
                video = mp.VideoFileClip(str(video_file))
                video.audio.write_audiofile(str(audio_file), verbose=False, logger=None)
                video.close()
                
            except Exception as e:
                print(f"❌ Error processing {video_file.name}: {e}")
        
        print(f"✓ Audio extraction complete")
        return True
    
    def extract_features_from_dataset(self, labels_file: str = None):
        """Extract features from all videos in the dataset"""
        
        # Check for your deception detection system
        try:
            # Import your feature extractors
            # Adjust this import to match your actual file structure
            from enhanced_deception_detector import (
                FacialAnalyzer,
                AudioAnalyzer,
                EnhancedLinguisticAnalyzer,
                EnhancedSpeechToTextConverter
            )
        except ImportError:
            print("❌ Could not import feature extractors.")
            print("Make sure your deception detection code is in the same directory")
            print("or adjust the import statement in this script.")
            return None
        
        # Initialize analyzers
        print("\nInitializing feature extractors...")
        facial = FacialAnalyzer()
        audio = AudioAnalyzer()
        linguistic = EnhancedLinguisticAnalyzer()
        speech = EnhancedSpeechToTextConverter()
        
        # Load labels
        if labels_file is None:
            labels_file = self.output_dir / 'labels.csv'
        
        if not Path(labels_file).exists():
            print(f"❌ Labels file not found: {labels_file}")
            print("Create a CSV file with columns: filename, label")
            print("Example:")
            print("  filename,label")
            print("  video1.mp4,0")
            print("  video2.mp4,1")
            return None
        
        df_labels = pd.read_csv(labels_file)
        
        # Process each video
        X_list = []
        y_list = []
        metadata_list = []
        
        print(f"\nProcessing {len(df_labels)} videos...")
        
        for idx, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc="Extracting features"):
            try:
                video_file = self.output_dir / 'raw_videos' / row['filename']
                audio_file = self.output_dir / 'audio_extracted' / f"{Path(row['filename']).stem}.wav"
                
                if not video_file.exists():
                    print(f"⚠ Video not found: {video_file}")
                    continue
                
                # Extract features
                facial_features = facial.extract_features(str(video_file))
                
                if audio_file.exists():
                    audio_features = audio.extract_features(str(audio_file))
                    text, disfluency = speech.convert_audio_to_text_with_fillers(str(audio_file))
                    linguistic_features = linguistic.extract_features(text, disfluency)
                else:
                    audio_features = {}
                    linguistic_features = {}
                
                # Combine features
                all_features = {
                    'facial': facial_features,
                    'audio': audio_features,
                    'linguistic': linguistic_features
                }
                
                # Convert to feature vector
                from trainable_model import TrainableDeceptionModel
                model = TrainableDeceptionModel()
                feature_vector = model.prepare_features(all_features)
                
                X_list.append(feature_vector)
                y_list.append(int(row['label']))
                metadata_list.append({
                    'filename': row['filename'],
                    'index': idx
                })
                
            except Exception as e:
                print(f"❌ Error processing {row['filename']}: {e}")
                continue
        
        if not X_list:
            print("❌ No features extracted")
            return None
        
        # Convert to arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Save features
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        features_file = self.features_dir / f'features_{timestamp}.npz'
        
        np.savez(features_file,
                 X=X,
                 y=y,
                 metadata=metadata_list,
                 feature_names=model.feature_names)
        
        print(f"\n✓ Features extracted and saved to: {features_file}")
        print(f"  Total samples: {len(X)}")
        print(f"  Features per sample: {X.shape[1]}")
        print(f"  Truthful: {np.sum(y==0)}, Deceptive: {np.sum(y==1)}")
        print(f"  Balance: {np.sum(y==1)/len(y)*100:.1f}% deceptive")
        
        return features_file
    
    def train_models(self, features_file: str = None):
        """Train models on extracted features"""
        
        from trainable_model import TrainableDeceptionModel
        
        # Find latest features file if not specified
        if features_file is None:
            feature_files = sorted(self.features_dir.glob('features_*.npz'))
            if not feature_files:
                print("❌ No feature files found. Run extraction first.")
                return
            features_file = feature_files[-1]
        
        print(f"\nLoading features from: {features_file}")
        data = np.load(features_file, allow_pickle=True)
        X = data['X']
        y = data['y']
        
        print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"Distribution: {np.sum(y==0)} truthful, {np.sum(y==1)} deceptive\n")
        
        # Train multiple models
        model_types = ['random_forest', 'gradient_boost', 'deep_learning']
        results = {}
        
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"Training: {model_type.upper().replace('_', ' ')}")
            print('='*60)
            
            model = TrainableDeceptionModel(model_type=model_type)
            val_results = model.train(X, y, validation_split=0.2)
            results[model_type] = val_results
            
            # Save model
            model_file = self.models_dir / f'model_{model_type}.pkl'
            model.save_model(str(model_file))
            
            # Show feature importance
            if model_type in ['random_forest', 'gradient_boost']:
                importance = model.get_feature_importance()
                print(f"\nTop 5 Important Features:")
                for i, (feat, score) in enumerate(list(importance.items())[:5], 1):
                    print(f"  {i}. {feat}: {score:.4f}")
        
        # Summary
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print('='*60)
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print('-'*60)
        
        for model_type, result in results.items():
            print(f"{model_type:<20} "
                  f"{result['accuracy']:<12.3f} "
                  f"{result['precision']:<12.3f} "
                  f"{result['recall']:<12.3f} "
                  f"{result['f1_score']:<12.3f}")
        
        print(f"\n✓ Models saved to: {self.models_dir}")
        
        # Save results summary
        summary_file = self.models_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            results_json = {}
            for model_type, result in results.items():
                results_json[model_type] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in result.items()
                    if k not in ['confusion_matrix', 'classification_report']
                }
            json.dump(results_json, f, indent=2)
        
        print(f"✓ Summary saved to: {summary_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for deception detection')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['setup', 'info', 'extract_audio', 'extract_features', 'train', 'all'],
                       help='Operation mode')
    parser.add_argument('--output_dir', type=str, default='training_data',
                       help='Output directory for all data')
    parser.add_argument('--labels', type=str, default=None,
                       help='Path to labels CSV file')
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(args.output_dir)
    
    if args.mode == 'info':
        preparer.create_sample_dataset_info()
    
    elif args.mode == 'setup':
        preparer.create_directory_structure()
        preparer.create_sample_dataset_info()
        print("\n✓ Setup complete!")
        print("\nNext steps:")
        print("1. Place your video files in training_data/raw_videos/")
        print("2. Create training_data/labels.csv with format:")
        print("     filename,label")
        print("     video1.mp4,0")
        print("     video2.mp4,1")
        print("3. Run: python prepare_training_data.py --mode extract_audio")
        print("4. Run: python prepare_training_data.py --mode extract_features")
        print("5. Run: python prepare_training_data.py --mode train")
    
    elif args.mode == 'extract_audio':
        print("\nExtracting audio from videos...")
        preparer.extract_audio_from_videos()
    
    elif args.mode == 'extract_features':
        print("\nExtracting features from dataset...")
        preparer.extract_features_from_dataset(args.labels)
    
    elif args.mode == 'train':
        print("\nTraining models...")
        preparer.train_models()
    
    elif args.mode == 'all':
        print("\n🚀 Running full pipeline...")
        preparer.create_directory_structure()
        print("\n" + "="*60)
        print("STEP 1: Extract Audio")
        print("="*60)
        preparer.extract_audio_from_videos()
        
        print("\n" + "="*60)
        print("STEP 2: Extract Features")
        print("="*60)
        features_file = preparer.extract_features_from_dataset(args.labels)
        
        if features_file:
            print("\n" + "="*60)
            print("STEP 3: Train Models")
            print("="*60)
            preparer.train_models(features_file)
            
            print("\n" + "="*70)
            print("🎉 COMPLETE! Your models are ready to use!")
            print("="*70)
            print(f"\nModels saved in: {preparer.models_dir}")
            print("\nTo use in your app:")
            print("1. Copy model files to your app directory")
            print("2. Launch the deception detection app")
            print("3. Select your trained model from the dropdown")
        else:
            print("\n❌ Feature extraction failed. Check logs above.")


if __name__ == "__main__":
    print("="*70)
    print("DECEPTION DETECTION - DATA PREPARATION TOOL")
    print("="*70)
    
    # Check if running with arguments
    if len(sys.argv) == 1:
        print("\nUsage examples:")
        print("  python prepare_training_data.py --mode setup")
        print("  python prepare_training_data.py --mode info")
        print("  python prepare_training_data.py --mode extract_audio")
        print("  python prepare_training_data.py --mode extract_features")
        print("  python prepare_training_data.py --mode train")
        print("  python prepare_training_data.py --mode all")
        print("\nOptions:")
        print("  --output_dir DIR    Output directory (default: training_data)")
        print("  --labels FILE       Path to labels CSV file")
        print("\nFor detailed help:")
        print("  python prepare_training_data.py --help")
        sys.exit(0)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
