import tkinter as tk
from tkinter import ttk, messagebox
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ================= LOAD MODEL =================
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    # If files aren't found, we'll show an error but keep the UI structure
    print(f"Model Error: {e}")

# ================= CLEAN TEXT =================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\(reuters\)|reuters', '', text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ================= GLOBAL STATS & KEYWORDS =================
stats = {"real": 0, "fake": 0, "uncertain": 0, "total": 0}

try:
    feature_names = vectorizer.get_feature_names_out()
    diff_scores = model.feature_log_prob_[0] - model.feature_log_prob_[1]
    top_fake_idx = np.argsort(diff_scores)[-150:]
    top_real_idx = np.argsort(diff_scores)[:150]
    FAKE_KEYWORDS = set(feature_names[i] for i in top_fake_idx)
    REAL_KEYWORDS = set(feature_names[i] for i in top_real_idx)
except:
    FAKE_KEYWORDS, REAL_KEYWORDS = set(), set()

# ================= LOGIC FUNCTIONS =================
def get_hits(text):
    words = text.lower().split()
    real_hits = [w for w in words if w in REAL_KEYWORDS]
    fake_hits = [w for w in words if w in FAKE_KEYWORDS]
    return list(set(real_hits)), list(set(fake_hits))

def draw_meter(value, label):
    color = "#3498db"
    if "REAL" in label: color = "#2ecc71"  # Emerald
    if "FAKE" in label: color = "#e74c3c"  # Alizarin Red
    if "UNCERTAIN" in label: color = "#f1c40f" # Sunflower Yellow

    # Facecolor matches the right panel background
    fig, ax = plt.subplots(figsize=(3.2, 2.2), facecolor='#1a1a2e')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)

    # Background Gauge
    theta = np.linspace(0, np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), color="#2d2d44", linewidth=15)

    # Active Gauge
    progress = (value / 100) * np.pi
    theta_val = np.linspace(np.pi, np.pi - progress, 100)
    ax.plot(np.cos(theta_val), np.sin(theta_val), color=color, linewidth=15, solid_capstyle='round')

    ax.text(0, 0.1, f"{value}%", color="white", fontsize=18, ha='center', fontweight='bold')
    ax.axis("off")
    return fig

# ================= MAIN ANALYZE =================
def analyze():
    raw_text = entry.get("1.0", tk.END).strip()
    if len(raw_text) < 10:
        messagebox.showwarning("Input Alert", "Please enter at least 10 characters.")
        return

    clean = clean_text(raw_text)
    vec = vectorizer.transform([clean])

    if vec.sum() == 0:
        result_var.set("⚠ OUT OF SCOPE")
        confidence_var.set("Confidence: 0%")
        suggestion_var.set("⚠️ ADVICE: Patterns unknown. Please verify manually!")
        update_meter(0, "UNCERTAIN")
        return

    proba = model.predict_proba(vec)[0]
    fake_p, real_p = proba[0], proba[1]
    real_hits, fake_hits = get_hits(clean)

    total_hits = len(real_hits) + len(fake_hits)
    kw_score = len(fake_hits) / total_hits if total_hits > 0 else 0.5
    final_fake = (fake_p * 0.7) + (kw_score * 0.3)
    final_real = 1 - final_fake
    diff = abs(final_real - final_fake)
    conf = round((diff * 100) ** 0.82, 2)

    stats["total"] += 1
    if diff < 0.20:
        label = "⚠ UNCERTAIN"
        color = "#f1c40f"
        stats["uncertain"] += 1
        suggestion = "⚠️ ADVICE: Be careful! The content contains mixed signals."
    elif final_real > final_fake:
        label = "🟢 REAL TWEET"
        color = "#2ecc71"
        stats["real"] += 1
        suggestion = "✅ ADVICE: Appears authentic, but cross-check with official news."
    else:
        label = "🔴 FAKE TWEET"
        color = "#e74c3c"
        stats["fake"] += 1
        suggestion = "🚫 ADVICE: High risk of misinformation! Do not share blindly."

    result_var.set(label)
    res_label.config(fg=color)
    confidence_var.set(f"Reliability Score: {conf}%")
    suggestion_var.set(suggestion)

    # Update bold stats
    stats_label.config(
        text=f"REAL: {stats['real']}  |  FAKE: {stats['fake']}  |  UNCERTAIN: {stats['uncertain']}  |  TOTAL: {stats['total']}"
    )
    update_meter(conf, label)

def update_meter(value, label):
    for w in meter_frame.winfo_children():
        w.destroy()
    fig = draw_meter(value, label)
    canvas = FigureCanvasTkAgg(fig, master=meter_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def clear():
    entry.delete("1.0", tk.END)
    result_var.set("")
    confidence_var.set("")
    suggestion_var.set("")
    for w in meter_frame.winfo_children():
        w.destroy()

# ================= PROFESSIONAL UI DESIGN =================
root = tk.Tk()
root.title("AI Fact-Checker Pro")
root.geometry("1100x700")  # FIXED: Removed (x)
root.configure(bg="#0f0c29") # Dark Purple-Blue Background

# Simulation of Gradient using frames
main_bg = tk.Frame(root, bg="#1a1a2e", padx=40, pady=40)
main_bg.place(relx=0, rely=0, relwidth=1, relheight=1)

# --- Header ---
header = tk.Label(main_bg, text="🛡️ AI TWEET VERIFICATION SYSTEM", font=("Helvetica", 24, "bold"), fg="#00d2ff", bg="#1a1a2e")
header.pack(pady=(0, 30))

content_frame = tk.Frame(main_bg, bg="#1a1a2e")
content_frame.pack(fill="both", expand=True)

# --- Left Panel ---
left_p = tk.Frame(content_frame, bg="#1a1a2e")
left_p.pack(side="left", fill="both", expand=True)

# Aligned Prompt and Input
tk.Label(left_p, text="Enter Content for Analysis:", font=("Verdana", 12, "bold"), fg="#9a8cff", bg="#1a1a2e").pack(anchor="w", pady=(0, 5))
entry = tk.Text(left_p, height=8, font=("Segoe UI", 12), bg="#2d2d44", fg="white", insertbackground="white", relief="flat", padx=15, pady=15)
entry.pack(fill="x", pady=(0, 15))

# Aligned Buttons
btn_p = tk.Frame(left_p, bg="#1a1a2e")
btn_p.pack(anchor="w")
tk.Button(btn_p, text="Verify Content", command=analyze, bg="#6a11cb", fg="white", font=("Arial", 11, "bold"), width=18, height=2, relief="flat", cursor="hand2").pack(side="left", padx=(0, 10))
tk.Button(btn_p, text="Clear Space", command=clear, bg="#444466", fg="white", font=("Arial", 11, "bold"), width=15, height=2, relief="flat", cursor="hand2").pack(side="left")

# Output Display Area
result_var, confidence_var, suggestion_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
res_label = tk.Label(left_p, textvariable=result_var, font=("Impact", 32), bg="#1a1a2e")
res_label.pack(anchor="w", pady=(30, 0))

tk.Label(left_p, textvariable=confidence_var, font=("Segoe UI", 14), fg="#00d2ff", bg="#1a1a2e").pack(anchor="w")

# Prominent Suggestion Text
tk.Label(left_p, textvariable=suggestion_var, font=("Segoe UI", 12, "bold italic"), fg="#ffcc00", bg="#1a1a2e", wraplength=500, justify="left").pack(anchor="w", pady=15)

# --- Right Panel (Stats & Meter) ---
right_p = tk.Frame(content_frame, bg="#1a1a2e", highlightbackground="#3a3a5a", highlightthickness=2, padx=20, pady=20)
right_p.pack(side="right", fill="y", padx=(30, 0))

tk.Label(right_p, text="SYSTEM STATUS", font=("Helvetica", 14, "bold"), fg="white", bg="#1a1a2e").pack(pady=(0, 10))

# BOLD STATS TEXT
stats_label = tk.Label(right_p, text="REAL: 0  |  FAKE: 0  |  UNCERTAIN: 0  |  TOTAL: 0",
                       fg="#00d2ff", bg="#1a1a2e", font=("Courier", 10, "bold"))
stats_label.pack(pady=10)

tk.Frame(right_p, height=1, bg="#3a3a5a", width=200).pack(pady=20)

tk.Label(right_p, text="CONFIDENCE GAUGE", font=("Helvetica", 10, "bold"), fg="#9a8cff", bg="#1a1a2e").pack()
meter_frame = tk.Frame(right_p, bg="#1a1a2e")
meter_frame.pack(pady=10)

root.mainloop()