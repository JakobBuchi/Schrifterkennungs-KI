import tkinter as tk
import numpy as np
import cv2
import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')  # Wichtig für GUI

# -------------------------
# MODELL LADEN
# -------------------------
try:
    model = tf.keras.models.load_model("buchstaben_ki_modell.h5")
    print("✅ Modell geladen!")
except:
    print("❌ Modell nicht gefunden! Erst Daten_weiterverarbeiten.py starten.")

# -------------------------
# GUI SETUP
# -------------------------
root = tk.Tk()
root.title("Buchstaben KI – PERFEKT")
root.geometry("1000x800")
root.configure(bg="#f0f0f0")

canvas = tk.Canvas(root, width=400, height=400, bg="white", bd=3, relief="ridge")
canvas.pack(pady=20)

canvas_array = np.ones((400, 400), dtype=np.uint8) * 255
last_x = None
last_y = None

# -------------------------
# ZEICHNEN (dicker Pinsel, sauber)
# -------------------------
def start_draw(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def paint(event):
    global last_x, last_y
    cv2.line(canvas_array, (int(last_x), int(last_y)), (event.x, event.y), 0, 15)  # Dicker!
    canvas.create_line(last_x, last_y, event.x, event.y, width=15, fill="black", capstyle=tk.ROUND)
    last_x, last_y = event.x, event.y

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", paint)

# -------------------------
# CANVAS LEEREN
# -------------------------
def clear_canvas():
    canvas.delete("all")
    canvas_array.fill(255)
    result_label.config(text="Canvas geleert – zeichne groß & mittig!")
    for w in chart_frame.winfo_children():
        w.destroy()

# -------------------------
# ERKENNUNG (VERBESSERT!)
# -------------------------
def erkennen():
    img = canvas_array.copy()
    
    # 1. Blur für saubere Kanten
    img = cv2.GaussianBlur(img, (3,3), 0)
    
    # 2. Buchstabenbereich extrahieren
    coords = np.column_stack(np.where(img < 200))
    if coords.size == 0:
        result_label.config(text="❌ Nichts gezeichnet!")
        return
    
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    img = img[y0:y1, x0:x1]
    
    # 3. Quadrat padding
    size = max(img.shape) + 10  # Etwas größer
    square = np.ones((size, size), dtype=np.uint8) * 255
    y_off = (size - img.shape[0]) // 2
    x_off = (size - img.shape[1]) // 2
    square[y_off:y_off+img.shape[0], x_off:x_off+img.shape[1]] = img
    
    # 4. Resize + INTER_AREA für bessere Qualität
    img = cv2.resize(square, (32, 32), interpolation=cv2.INTER_AREA)
    
    # 5. Normalisieren (EXAKT wie Training!)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 32, 32, 1)
    
    # 6. Vorhersage
    pred = model.predict(img, verbose=0)[0]
    idx = np.argmax(pred)
    letter = chr(65 + idx)  # A=0, B=1...
    conf = pred[idx]
    
    result_label.config(text=f"🎯 Erkannt: '{letter}' (Confidence: {conf:.1%})")
    
    # 7. Balkendiagramm
    fig, ax = plt.subplots(figsize=(10,4))
    letters = [chr(65+i) for i in range(26)]
    bars = ax.bar(letters, pred, color='skyblue', alpha=0.7)
    ax.set_title("Wahrscheinlichkeiten A-Z", fontsize=14)
    ax.set_ylabel("Confidence")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Top-5 hervorheben
    top_indices = np.argsort(pred)[-5:][::-1]
    for i in top_indices:
        bars[i].set_color('red' if i==idx else 'orange')
    
    for w in chart_frame.winfo_children():
        w.destroy()
    canvas_plot = FigureCanvasTkAgg(fig, chart_frame)
    canvas_plot.draw()
    canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# -------------------------
# BUTTONS
# -------------------------
btn_frame = tk.Frame(root, bg="#f0f0f0")
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="🔍 ERKENNEN", command=erkennen, 
          font=("Arial", 16, "bold"), bg="#4CAF50", fg="white", width=12).pack(side=tk.LEFT, padx=10)
tk.Button(btn_frame, text="🗑️ LEEREN", command=clear_canvas, 
          font=("Arial", 16, "bold"), bg="#f44336", fg="white", width=12).pack(side=tk.LEFT, padx=10)

# -------------------------
# INFO + CHART
# -------------------------
info_label = tk.Label(root, text="💡 Tipp: Zeichne GROSS & DICHT in der MITTE!", 
                      font=("Arial", 12), bg="#f0f0f0")
info_label.pack(pady=5)

result_label = tk.Label(root, text="Zeichne einen Buchstaben (A-Z)", 
                        font=("Arial", 20, "bold"), bg="#f0f0f0")
result_label.pack(pady=10)

chart_frame = tk.Frame(root, bg="white", relief="sunken", bd=2)
chart_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

# -------------------------
# START
# -------------------------
root.mainloop()
