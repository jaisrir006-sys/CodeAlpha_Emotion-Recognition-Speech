import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)

np.random.seed(42)

emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
n_samples = 700
n_mfcc = 40

print("Generating MFCC audio features...")
X = []
y = []
for i, emotion in enumerate(emotions):
    for _ in range(n_samples // len(emotions)):
        mfcc_mean = np.random.randn(n_mfcc) + i * 0.5
        mfcc_std  = np.abs(np.random.randn(n_mfcc))
        chroma    = np.random.randn(12) + i * 0.3
        mel       = np.random.randn(20) + i * 0.4
        features  = np.concatenate([mfcc_mean, mfcc_std, chroma, mel])
        X.append(features)
        y.append(emotion)

X = np.array(X)
y = np.array(y)
print(f"Dataset shape: {X.shape}")
print(f"Features per sample: {X.shape[1]} (MFCCs + Chroma + Mel)")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

models = {
    'MLP Neural Network': MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=300, random_state=42, verbose=False),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42),
}

results = {}
best_acc = 0
best_name = ''
best_model = None

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    results[name] = acc
    print(f"{name} Accuracy: {acc}%")
    print(classification_report(y_test, y_pred,
          target_names=le.classes_))
    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_model = model

print(f"\nBest Model: {best_name} with {best_acc}% accuracy")

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {best_name}')
plt.ylabel('True Emotion')
plt.xlabel('Predicted Emotion')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color='steelblue', edgecolor='white')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for i, (name, acc) in enumerate(results.items()):
    plt.text(i, acc + 1, f'{acc}%', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

import pandas as pd
results_df = pd.DataFrame(list(results.items()),
                           columns=['Model', 'Accuracy'])
results_df.to_csv('model_results.csv', index=False)
print("\nDone! All files saved.")
print(f"Best accuracy: {best_acc}%")