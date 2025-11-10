import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
from typing import Dict, Tuple, List
import os


class MultimodalDeceptionClassifier(nn.Module):
    """
    Deep learning model that learns to combine multimodal features
    for deception detection.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(MultimodalDeceptionClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class AttentionFusionModel(nn.Module):
    """
    Advanced model with attention mechanism to learn importance
    of different modalities (facial, audio, linguistic).
    """
    
    def __init__(self, facial_dim: int, audio_dim: int, linguistic_dim: int):
        super(AttentionFusionModel, self).__init__()
        
        # Separate encoders for each modality
        self.facial_encoder = nn.Sequential(
            nn.Linear(facial_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.linguistic_encoder = nn.Sequential(
            nn.Linear(linguistic_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, facial, audio, linguistic):
        # Encode each modality
        facial_encoded = self.facial_encoder(facial)
        audio_encoded = self.audio_encoder(audio)
        linguistic_encoded = self.linguistic_encoder(linguistic)
        
        # Concatenate for attention
        combined = torch.cat([facial_encoded, audio_encoded, linguistic_encoded], dim=1)
        
        # Calculate attention weights
        attention_weights = self.attention(combined)
        
        # Apply attention weights
        weighted_facial = facial_encoded * attention_weights[:, 0].unsqueeze(1)
        weighted_audio = audio_encoded * attention_weights[:, 1].unsqueeze(1)
        weighted_linguistic = linguistic_encoded * attention_weights[:, 2].unsqueeze(1)
        
        # Fuse modalities
        fused = weighted_facial + weighted_audio + weighted_linguistic
        
        # Final classification
        output = self.classifier(fused)
        
        return output, attention_weights


class TrainableDeceptionModel:
    """
    Wrapper class that provides multiple ML model options with training,
    evaluation, and deployment capabilities.
    """
    
    def __init__(self, model_type: str = 'deep_learning'):
        """
        Args:
            model_type: 'deep_learning', 'attention_fusion', 'random_forest', 
                       'gradient_boost', or 'svm'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.trained = False
    
    def prepare_features(self, features_dict: Dict) -> np.ndarray:
        """Convert multimodal features dictionary to feature vector."""
        feature_vector = []
        feature_names = []
        
        # Facial features
        facial = features_dict.get('facial', {})
        facial_features = [
            facial.get('blink_rate', 0),
            np.mean(facial.get('eye_aspect_ratios', [0])),
            np.std(facial.get('eye_aspect_ratios', [0])),
            np.mean(facial.get('mouth_movement', [0])),
            np.mean(facial.get('micro_expressions', [0])),
        ]
        feature_vector.extend(facial_features)
        feature_names.extend(['blink_rate', 'ear_mean', 'ear_std', 
                            'mouth_movement', 'micro_expressions'])
        
        # Audio features
        audio = features_dict.get('audio', {})
        audio_features = [
            audio.get('pitch_mean', 0),
            audio.get('pitch_std', 0),
            audio.get('pitch_range', 0),
            audio.get('speech_rate', 0),
            audio.get('pause_ratio', 0),
            audio.get('energy_mean', 0),
            audio.get('energy_std', 0),
            audio.get('spectral_centroid_mean', 0),
            audio.get('zcr_mean', 0),
        ]
        feature_vector.extend(audio_features)
        feature_names.extend(['pitch_mean', 'pitch_std', 'pitch_range',
                            'speech_rate', 'pause_ratio', 'energy_mean',
                            'energy_std', 'spectral_centroid', 'zcr_mean'])
        
        # Linguistic features
        linguistic = features_dict.get('linguistic', {})
        linguistic_features = [
            linguistic.get('word_count', 0),
            linguistic.get('filler_words_ratio', 0),
            linguistic.get('uncertainty_words_ratio', 0),
            linguistic.get('certainty_words_ratio', 0),
            linguistic.get('negation_words_ratio', 0),
            linguistic.get('personal_pronouns_ratio', 0),
            linguistic.get('cognitive_words_ratio', 0),
            linguistic.get('repetition_ratio', 0),
            linguistic.get('pause_ratio', 0),
        ]
        feature_vector.extend(linguistic_features)
        feature_names.extend(['word_count', 'filler_ratio', 'uncertainty_ratio',
                            'certainty_ratio', 'negation_ratio', 'pronoun_ratio',
                            'cognitive_ratio', 'repetition_ratio', 'text_pause_ratio'])
        
        if not self.feature_names:
            self.feature_names = feature_names
        
        return np.array(feature_vector)
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """
        Train the model on labeled data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - 0 for truth, 1 for deception
            validation_split: Fraction of data to use for validation
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        if self.model_type == 'deep_learning':
            self._train_deep_learning(X_train_scaled, y_train, X_val_scaled, y_val)
        
        elif self.model_type == 'attention_fusion':
            self._train_attention_model(X_train_scaled, y_train, X_val_scaled, y_val)
        
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
        
        elif self.model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                subsample=0.8,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
        
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
        
        self.trained = True
        
        # Evaluate on validation set
        val_results = self.evaluate(X_val, y_val)
        print(f"\nValidation Results for {self.model_type}:")
        print(f"Accuracy: {val_results['accuracy']:.3f}")
        print(f"Precision: {val_results['precision']:.3f}")
        print(f"Recall: {val_results['recall']:.3f}")
        print(f"F1-Score: {val_results['f1_score']:.3f}")
        print(f"AUC-ROC: {val_results['auc_roc']:.3f}")
        
        return val_results
    
    def _train_deep_learning(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train deep learning model."""
        input_dim = X_train.shape[1]
        self.model = MultimodalDeceptionClassifier(input_dim)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def _train_attention_model(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train attention fusion model."""
        # Split features into modalities
        # Assuming first 5 are facial, next 9 are audio, rest are linguistic
        facial_dim = 5
        audio_dim = 9
        linguistic_dim = X_train.shape[1] - facial_dim - audio_dim
        
        self.model = AttentionFusionModel(facial_dim, audio_dim, linguistic_dim)
        
        # Prepare tensors
        X_train_facial = torch.FloatTensor(X_train[:, :facial_dim])
        X_train_audio = torch.FloatTensor(X_train[:, facial_dim:facial_dim+audio_dim])
        X_train_ling = torch.FloatTensor(X_train[:, facial_dim+audio_dim:])
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        X_val_facial = torch.FloatTensor(X_val[:, :facial_dim])
        X_val_audio = torch.FloatTensor(X_val[:, facial_dim:facial_dim+audio_dim])
        X_val_ling = torch.FloatTensor(X_val[:, facial_dim+audio_dim:])
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            outputs, attention_weights = self.model(X_train_facial, X_train_audio, X_train_ling)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs, val_attention = self.model(X_val_facial, X_val_audio, X_val_ling)
                val_loss = criterion(val_outputs, y_val_tensor)
            
            if (epoch + 1) % 10 == 0:
                avg_attention = val_attention.mean(dim=0)
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
                print(f"  Attention weights - Facial: {avg_attention[0]:.3f}, "
                      f"Audio: {avg_attention[1]:.3f}, Linguistic: {avg_attention[2]:.3f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def predict(self, features_dict: Dict) -> Dict:
        """
        Predict deception probability for new sample.
        
        Returns:
            Dictionary with 'probability', 'prediction', and 'confidence'
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare features
        X = self.prepare_features(features_dict).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        if self.model_type in ['deep_learning', 'attention_fusion']:
            self.model.eval()
            with torch.no_grad():
                if self.model_type == 'deep_learning':
                    X_tensor = torch.FloatTensor(X_scaled)
                    prob = self.model(X_tensor).item()
                else:
                    # Split into modalities for attention model
                    facial_dim = 5
                    audio_dim = 9
                    X_facial = torch.FloatTensor(X_scaled[:, :facial_dim])
                    X_audio = torch.FloatTensor(X_scaled[:, facial_dim:facial_dim+audio_dim])
                    X_ling = torch.FloatTensor(X_scaled[:, facial_dim+audio_dim:])
                    prob, attention = self.model(X_facial, X_audio, X_ling)
                    prob = prob.item()
        else:
            prob = self.model.predict_proba(X_scaled)[0][1]
        
        prediction = 1 if prob > 0.5 else 0
        confidence = abs(prob - 0.5) * 2
        
        return {
            'probability': prob,
            'prediction': prediction,
            'prediction_label': 'Deception' if prediction == 1 else 'Truth',
            'confidence': confidence
        }
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance."""
        X_scaled = self.scaler.transform(X)
        
        if self.model_type in ['deep_learning', 'attention_fusion']:
            self.model.eval()
            with torch.no_grad():
                if self.model_type == 'deep_learning':
                    X_tensor = torch.FloatTensor(X_scaled)
                    y_pred_proba = self.model(X_tensor).numpy().flatten()
                else:
                    facial_dim = 5
                    audio_dim = 9
                    X_facial = torch.FloatTensor(X_scaled[:, :facial_dim])
                    X_audio = torch.FloatTensor(X_scaled[:, facial_dim:facial_dim+audio_dim])
                    X_ling = torch.FloatTensor(X_scaled[:, facial_dim+audio_dim:])
                    y_pred_proba, _ = self.model(X_facial, X_audio, X_ling)
                    y_pred_proba = y_pred_proba.numpy().flatten()
        else:
            y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        auc_roc = roc_auc_score(y, y_pred_proba)
        
        return {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'auc_roc': auc_roc,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance (for tree-based models)."""
        if self.model_type not in ['random_forest', 'gradient_boost']:
            return None
        
        importances = self.model.feature_importances_
        feature_importance = {
            name: importance 
            for name, importance in zip(self.feature_names, importances)
        }
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features)
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        model_data = {
            'model_type': self.model_type,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'trained': self.trained
        }
        
        if self.model_type in ['deep_learning', 'attention_fusion']:
            model_data['model_state'] = self.model.state_dict()
            model_data['model_architecture'] = {
                'type': self.model_type,
                'input_dim': len(self.feature_names)
            }
        else:
            model_data['model'] = self.model
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_type = model_data['model_type']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.trained = model_data['trained']
        
        if self.model_type in ['deep_learning', 'attention_fusion']:
            if self.model_type == 'deep_learning':
                self.model = MultimodalDeceptionClassifier(
                    model_data['model_architecture']['input_dim']
                )
            else:
                # Reconstruct attention model
                self.model = AttentionFusionModel(5, 9, 
                    model_data['model_architecture']['input_dim'] - 14)
            
            self.model.load_state_dict(model_data['model_state'])
            self.model.eval()
        else:
            self.model = model_data['model']
        
        print(f"Model loaded from {filepath}")


# Example usage and data generation for demonstration
def generate_synthetic_training_data(n_samples=1000):
    """
    Generate synthetic training data for demonstration.
    In practice, you would use real labeled data from psychological studies.
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    for i in range(n_samples):
        is_deceptive = np.random.rand() > 0.5
        
        if is_deceptive:
            # Deceptive patterns (simplified)
            features = {
                'facial': {
                    'blink_rate': np.random.normal(25, 5),
                    'eye_aspect_ratios': [np.random.normal(0.25, 0.05) for _ in range(10)],
                    'mouth_movement': [np.random.normal(0.3, 0.1) for _ in range(10)],
                    'micro_expressions': [np.random.normal(0.15, 0.05) for _ in range(10)],
                },
                'audio': {
                    'pitch_mean': np.random.normal(150, 20),
                    'pitch_std': np.random.normal(60, 10),
                    'pitch_range': np.random.normal(120, 20),
                    'speech_rate': np.random.normal(2.0, 0.5),
                    'pause_ratio': np.random.normal(0.35, 0.1),
                    'energy_mean': np.random.normal(0.4, 0.1),
                    'energy_std': np.random.normal(0.15, 0.05),
                    'spectral_centroid_mean': np.random.normal(2000, 300),
                    'zcr_mean': np.random.normal(0.08, 0.02),
                },
                'linguistic': {
                    'word_count': np.random.randint(50, 200),
                    'filler_words_ratio': np.random.normal(0.08, 0.02),
                    'uncertainty_words_ratio': np.random.normal(0.05, 0.02),
                    'certainty_words_ratio': np.random.normal(0.02, 0.01),
                    'negation_words_ratio': np.random.normal(0.04, 0.02),
                    'personal_pronouns_ratio': np.random.normal(0.06, 0.02),
                    'cognitive_words_ratio': np.random.normal(0.03, 0.01),
                    'repetition_ratio': np.random.normal(0.02, 0.01),
                    'pause_ratio': np.random.normal(0.03, 0.01),
                }
            }
        else:
            # Truthful patterns
            features = {
                'facial': {
                    'blink_rate': np.random.normal(15, 3),
                    'eye_aspect_ratios': [np.random.normal(0.28, 0.03) for _ in range(10)],
                    'mouth_movement': [np.random.normal(0.15, 0.05) for _ in range(10)],
                    'micro_expressions': [np.random.normal(0.08, 0.03) for _ in range(10)],
                },
                'audio': {
                    'pitch_mean': np.random.normal(130, 15),
                    'pitch_std': np.random.normal(35, 8),
                    'pitch_range': np.random.normal(80, 15),
                    'speech_rate': np.random.normal(2.5, 0.3),
                    'pause_ratio': np.random.normal(0.15, 0.05),
                    'energy_mean': np.random.normal(0.5, 0.1),
                    'energy_std': np.random.normal(0.1, 0.03),
                    'spectral_centroid_mean': np.random.normal(1800, 200),
                    'zcr_mean': np.random.normal(0.06, 0.01),
                },
                'linguistic': {
                    'word_count': np.random.randint(80, 250),
                    'filler_words_ratio': np.random.normal(0.02, 0.01),
                    'uncertainty_words_ratio': np.random.normal(0.015, 0.01),
                    'certainty_words_ratio': np.random.normal(0.04, 0.015),
                    'negation_words_ratio': np.random.normal(0.02, 0.01),
                    'personal_pronouns_ratio': np.random.normal(0.08, 0.02),
                    'cognitive_words_ratio': np.random.normal(0.05, 0.015),
                    'repetition_ratio': np.random.normal(0.005, 0.003),
                    'pause_ratio': np.random.normal(0.015, 0.005),
                }
            }
        
        model = TrainableDeceptionModel()
        X.append(model.prepare_features(features))
        y.append(1 if is_deceptive else 0)
    
    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("=== Trainable Deception Detection Models ===\n")
    
    # Generate synthetic training data
    print("Generating synthetic training data...")
    X, y = generate_synthetic_training_data(n_samples=1000)
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y==0)} truthful, {np.sum(y==1)} deceptive\n")
    
    # Train and compare different models
    models_to_test = ['random_forest', 'gradient_boost', 'deep_learning']
    
    results = {}
    
    for model_type in models_to_test:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper().replace('_', ' ')} model...")
        print('='*60)
        
        model = TrainableDeceptionModel(model_type=model_type)
        val_results = model.train(X, y, validation_split=0.2)
        results[model_type] = val_results
        
        # Show feature importance for tree-based models
        if model_type in ['random_forest', 'gradient_boost']:
            importance = model.get_feature_importance()
            print(f"\nTop 5 Most Important Features:")
            for i, (feature, score) in enumerate(list(importance.items())[:5], 1):
                print(f"  {i}. {feature}: {score:.4f}")
        
        # Save model
        model.save_model(f'deception_model_{model_type}.pkl')
    
    # Compare all models
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print('='*60)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10}")
    print('-'*60)
    
    for model_type, result in results.items():
        print(f"{model_type:<20} {result['accuracy']:<10.3f} {result['precision']:<10.3f} "
              f"{result['recall']:<10.3f} {result['f1_score']:<10.3f} {result['auc_roc']:<10.3f}")
    
    print("\n✓ Models trained and saved successfully!")
    print("\nTo use in your application:")
    print("1. Collect real labeled data from validated studies")
    print("2. Train the model: model.train(X, y)")
    print("3. Save: model.save_model('my_model.pkl')")
    print("4. Load in your app: model.load_model('my_model.pkl')")
    print("5. Predict: result = model.predict(features_dict)")
