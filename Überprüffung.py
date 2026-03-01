import numpy as np
import tensorflow as tf

# Modell und Testdaten laden
model = tf.keras.models.load_model('buchstaben_ki_modell.h5')
X_test = np.load('train_images.npy')  # Vollständige Daten laden
y_test_full = np.load('train_labels.npy')

# 10% Test-Split wie beim Training (gleicher random_state für Reproduzierbarkeit)
from sklearn.model_selection import train_test_split
X_test_split, _, y_test_split, _ = train_test_split(X_test, y_test_full, test_size=0.1, random_state=42, stratify=y_test_full)

# Auf CNN-Shape bringen (32,32,1)
X_test_split = X_test_split.reshape(-1, 32, 32, 1).astype('float32')

print(f"Testdaten: {X_test_split.shape[0]} Bilder")

# 1. Funktion: Vorhersagen → Buchstaben rückberechnen
def predictions_to_letters(predictions, num_classes=26):
    """
    predictions: Shape (N, 26) - Wahrscheinlichkeiten pro Klasse
    Rückgabe: Liste mit vorhergesagten Buchstaben 'A', 'B', ..., 'Z'
    """
    pred_indices = np.argmax(predictions, axis=1)  # Index der max. Wahrscheinlichkeit
    letters = [chr(65 + idx) for idx in pred_indices]  # 0→'A', 1→'B', ..., 25→'Z'
    return letters, pred_indices

# 2. Vorhersagen generieren
print("Generiere Vorhersagen...")
predictions = model.predict(X_test_split, verbose=0)  # Shape: (N_test, 26)

# 3. Vorhergesagte Labels berechnen
predicted_letters, predicted_indices = predictions_to_letters(predictions)
true_indices = y_test_split.flatten()
true_letters = [chr(65 + idx) for idx in true_indices]

# 4. Manuelle Überprüfung: Korrektheit prüfen
correct = sum(1 for p, t in zip(predicted_indices, true_indices) if p == t)
accuracy = correct / len(predicted_indices)

print("\n" + "="*60)
print("MANUELLE NETZWERK-ÜBERPRÜFUNG")
print("="*60)
print(f"Anzahl Testbilder: {len(predicted_letters)}")
print(f"Korrekt vorhergesagt: {correct}/{len(predicted_letters)}")
print(f"Genauigkeit: {accuracy:.1%} ({accuracy:.4f})")
print("\nErste 20 Beispiele:")
print("Bild-Nr | Wahres Label | Vorhergesagt | Korrekt?")
print("-" * 50)
for i in range(min(20, len(predicted_letters))):
    is_correct = "✅" if predicted_indices[i] == true_indices[i] else "❌"
    print(f"{i+1:7d} |     {true_letters[i]:<2}     |     {predicted_letters[i]:<2}     | {is_correct}")

# 5. Alle falsch vorhergesagte anzeigen (optional)
false_preds = [(i, true_letters[i], predicted_letters[i]) for i in range(len(predicted_letters)) 
               if predicted_indices[i] != true_indices[i]]
print(f"\nFalsch vorhergesagt: {len(false_preds)}/{len(predicted_letters)}")
if false_preds:
    print("Erste 10 Fehler (Nr, Wahr, Vorhergesagt):")
    for nr, wahr, vor in false_preds[:10]:
        print(f"  {nr}: {wahr} → {vor}")
