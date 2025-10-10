import numpy as np
from trainable_model import TrainableDeceptionModel
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def train_on_dolos_data():
    """Train models using processed DOLOS dataset"""
    
    print("=" * 70)
    print("TRAINING WITH DOLOS DATASET")
    print("=" * 70)
    
    # Load processed features
    print("\nLoading processed DOLOS features...")
    data = np.load('dolos_features.npz', allow_pickle=True)
    X = data['X']
    y = data['y']
    metadata = data['metadata']
    
    print(f"Loaded: {len(X)} samples, {X.shape[1]} features")
    print(f"Distribution: {np.sum(y==0)} truthful, {np.sum(y==1)} deceptive")
    print(f"Balance: {np.sum(y==1)/len(y)*100:.1f}% deceptive")
    
    # Handle class imbalance if needed
    if abs(np.sum(y==0) - np.sum(y==1)) > len(y) * 0.2:
        print("\n⚠ Dataset is imbalanced. Consider using class weights.")
    
    # Train multiple models
    models = {
        'random_forest': TrainableDeceptionModel('random_forest'),
        'gradient_boost': TrainableDeceptionModel('gradient_boost'),
        'deep_learning': TrainableDeceptionModel('deep_learning'),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name.upper().replace('_', ' ')}")
        print('='*60)
        
        # Train with cross-validation
        val_results = model.train(X, y, validation_split=0.2)
        results[name] = val_results
        
        # Perform 5-fold cross-validation
        if name in ['random_forest', 'gradient_boost']:
            print("\nPerforming 5-fold cross-validation...")
            cv_scores = cross_val_score(model.model, X, y, cv=5, scoring='accuracy')
            print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Save model
        model.save_model(f'dolos_model_{name}.pkl')
        
        # Feature importance
        if name in ['random_forest', 'gradient_boost']:
            importance = model.get_feature_importance()
            print(f"\nTop 10 Most Important Features:")
            for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
                print(f"  {i:2d}. {feature:30s}: {score:.4f}")
    
    # Compare models
    print(f"\n{'='*70}")
    print("MODEL COMPARISON ON DOLOS DATASET")
    print('='*70)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print('-'*70)
    
    for name, result in results.items():
        print(f"{name:<20} "
              f"{result['accuracy']:<12.3f} "
              f"{result['precision']:<12.3f} "
              f"{result['recall']:<12.3f} "
              f"{result['f1_score']:<12.3f} "
              f"{result['auc_roc']:<12.3f}")
    
    # Plot comparison
    plot_results(results)
    
    print("\n✓ Training complete! Models saved with 'dolos_model_' prefix")
    
    return results

def plot_results(results):
    """Visualize model comparison"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (name, result) in enumerate(results.items()):
        values = [result[m] for m in metrics]
        ax.bar(x + i*width, values, width, label=name.replace('_', ' ').title())
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison on DOLOS Dataset')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dolos_model_comparison.png', dpi=300)
    print("\n✓ Comparison plot saved: dolos_model_comparison.png")

if __name__ == "__main__":
    results = train_on_dolos_data()
