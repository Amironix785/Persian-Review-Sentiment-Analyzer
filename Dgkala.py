import tkinter as tk
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

data = pd.read_csv("Dgkala_train.csv")

x = data["Text"]
y = data["Suggestion"]

tf = TfidfVectorizer()
x_tf = tf.fit_transform(x)

model = LinearSVC()
model.fit(x_tf, y)


root = tk.Tk()
root.title("نظر سنجی هوشمند")
root.geometry("600x500")
root.configure(bg="#101827")
root.resizable(False, False)

text = tk.Text(root, font=("Tahoma", 14), bg="#1e293b", fg="white",
               insertbackground="white", relief="flat", height=8)
text.pack(padx=40, pady=(50, 20), fill="x")

def predict():
    value = text.get("1.0", "end").strip()

    if not value:
        return

    suggestion = model.predict(tf.transform([value]))[0]

    names = {
        1: "عالی بود 😊",
        2: "بی‌تفاوت 😐",
        3: "افتضاح بود 😞"
    }

    score = round(data[data["Suggestion"] == suggestion]["Score"].mean())

    for w in root.winfo_children():
        w.destroy()

    tk.Label(root, text="نتیجه تحلیل نظر",
             font=("Tahoma", 24, "bold"),
             bg="#101827", fg="white").pack(pady=70)

    tk.Label(root, text=names[suggestion],
             font=("Tahoma", 25, "bold"),
             bg="#101827", fg="#22c55e").pack(pady=15)

    tk.Label(root, text=f"Score: {score}",
             font=("Tahoma", 18),
             bg="#101827", fg="#38bdf8").pack(pady=10)

    tk.Button(root, text="← نظر جدید", command=main,
              font=("Tahoma", 12), bg="#334155", fg="white",
              relief="flat", padx=30, pady=10).pack(pady=40)

def main():
    for w in root.winfo_children():
        w.destroy()

    tk.Label(root, text="🤖 تحلیل نظر کاربران",
             font=("Tahoma", 24, "bold"),
             bg="#101827", fg="white").pack(pady=(35, 10))

    tk.Label(root, text="نظر خود را درباره محصول بنویسید",
             font=("Tahoma", 12),
             bg="#101827", fg="#94a3b8").pack(pady=(0, 20))

    global text
    text = tk.Text(root, font=("Tahoma", 14), bg="#1e293b",
                   fg="white", insertbackground="white",
                   relief="flat", height=8)
    text.pack(padx=40, pady=10, fill="x")

    tk.Button(root, text="🔍 تحلیل نظر",
              command=predict, font=("Tahoma", 13, "bold"),
              bg="#38bdf8", fg="#0f172a",
              relief="flat", padx=35, pady=10).pack(pady=25)

main()
root.mainloop()