import numpy as np
from trainable_model import TrainableDeceptionModel, generate_synthetic_training_data
import sys

def main():
    print("=" * 70)
    print("DECEPTION DETECTION MODEL TRAINING")
    print("=" * 70)
    print()
    
    # In production, load your real labeled dataset here
    print("Loading training data...")
    print("Note: Using synthetic data for demonstration.")
    print("Replace with real labeled data from psychological studies.")
    print()
    
    X, y = generate_synthetic_training_data(n_samples=2000)
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Classes: {np.sum(y==0)} truthful, {np.sum(y==1)} deceptive")
    print()
    
    # Train multiple models
    model_types = ['random_forest', 'gradient_boost', 'deep_learning']
    
    print("Which model would you like to train?")
    print("1. Random Forest (Fast, interpretable)")
    print("2. Gradient Boosting (High accuracy)")
    print("3. Deep Learning (Best for large datasets)")
    print("4. All models")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        model_types = ['random_forest']
    elif choice == '2':
        model_types = ['gradient_boost']
    elif choice == '3':
        model_types = ['deep_learning']
    elif choice == '4':
        pass
    else:
        print("Invalid choice. Training all models.")
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper().replace('_', ' ')}")
        print('='*60)
        
        model = TrainableDeceptionModel(model_type=model_type)
        results = model.train(X, y, validation_split=0.2)
        
        # Save model
        filename = f'deception_model_{model_type}.pkl'
        model.save_model(filename)
        
        print(f"\n✓ Model saved as: {filename}")
        print(f"Validation Accuracy: {results['accuracy']:.3f}")
        print(f"AUC-ROC: {results['auc_roc']:.3f}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print('='*60)
    print("\nModels are ready to use in the detection application.")
    print("Restart the app to see the new models in the dropdown menu.")

if __name__ == "__main__":
    main()
