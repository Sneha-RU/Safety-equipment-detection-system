import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
from ultralytics import YOLO
import os
import glob
import cv2
import tempfile

CUSTOM_NAMES = ['Hardhat', 'NO-Hardhat', 'Safety Vest', 'NO-Safety Vest', 'Person']

BG_MAIN = "#0D1117"
BG_PANEL = "#161B22"
ACCENT = "#00FF88"
TEXT_MAIN = "#F0F6FC"
HEADER_FONT = ("Montserrat", 18, "bold")
BODY_FONT = ("Montserrat", 12)
BTN_FONT = ("Montserrat", 12)
FOOTER_FONT = ("Montserrat", 9, "italic")

model = YOLO(r"best.pt")   

root = tk.Tk()
root.title("PPE Compliance Detector")
root.geometry("900x750")
root.configure(bg=BG_MAIN)

main_container = tk.Frame(root, bg=BG_MAIN)
main_container.pack(fill="both", expand=True, padx=30, pady=20)

header = tk.Frame(main_container, bg=BG_MAIN)
header.pack(fill="x", pady=(0, 30))
tk.Label(header, text="🦺 PPE COMPLIANCE DETECTOR 🦺", font=("Montserrat", 24, "bold"), fg=ACCENT, bg=BG_MAIN).pack()
tk.Label(header, text="SAFETY EQUIPMENT DETECTION SYSTEM", font=("Montserrat", 16, "bold"), fg="#FF6B9D", bg=BG_MAIN).pack(pady=(5, 0))
tk.Label(header, text="Real-time hardhat & safety vest compliance monitoring", font=("Montserrat", 11), fg="#B084CC", bg=BG_MAIN).pack(pady=(8, 0))

content_frame = tk.Frame(main_container, bg=BG_MAIN)
content_frame.pack(fill="both", expand=True, pady=(0, 20))

left_panel = tk.Frame(content_frame, bg=BG_MAIN)
left_panel.pack(side="left", fill="both", expand=True, padx=(0, 15))

img_panel = tk.Frame(left_panel, bg=BG_PANEL, bd=2, relief=tk.SOLID, highlightbackground=ACCENT, highlightthickness=1)
img_panel.pack(fill="both", expand=True, pady=(0, 15))
img_label = tk.Label(img_panel, bg=BG_PANEL, bd=0, text="🦺 DETECTION AREA 🦺", font=("Montserrat", 14, "bold"), fg="#666")
img_label.pack(expand=True)

right_panel = tk.Frame(content_frame, bg=BG_MAIN, width=300)
right_panel.pack(side="right", fill="y", padx=(15, 0))
right_panel.pack_propagate(False)

result_panel = tk.Frame(right_panel, bg=BG_PANEL, bd=2, relief=tk.SOLID, highlightbackground=ACCENT, highlightthickness=1)
result_panel.pack(fill="both", expand=True)

result_header = tk.Frame(result_panel, bg=BG_PANEL)
result_header.pack(fill="x", padx=15, pady=(15, 10))
tk.Label(result_header, text="🎯 DETECTION RESULTS", font=("Montserrat", 12, "bold"), fg=ACCENT, bg=BG_PANEL).pack()

output_text = tk.Text(result_panel, height=15, bg=BG_PANEL, fg=TEXT_MAIN, font=BODY_FONT, bd=0, relief=tk.FLAT, highlightthickness=0, wrap=tk.WORD)
output_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
output_text.config(state="disabled")

live_running = False
cap_live = None

def clear_previous_outputs(folder="runs/detect/predict"):
    if os.path.exists(folder):
        files = glob.glob(os.path.join(folder, "*"))
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass

def display_result_image(result_path):
    try:
        img = Image.open(result_path)
        img = ImageOps.pad(img, (430, 310), color=BG_PANEL)
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk
    except Exception:
        output_text.insert(tk.END, "Error loading image.\n")
        output_text.config(state="disabled")

def handle_detection_results(results):
    output_text.config(state="normal")
    output_text.delete(1.0, tk.END)
    if results[0].boxes is not None and len(results[0].boxes.cls) > 0:
        class_indices = [int(x) for x in results[0].boxes.cls]
        detected = []
        for i in class_indices:
            if 0 <= i < len(CUSTOM_NAMES):
                detected.append(CUSTOM_NAMES[i])
            else:
                detected.append(f"Unknown({i})")
        summary = {cls: detected.count(cls) for cls in CUSTOM_NAMES}
        for key in summary:
            if summary[key]:
                output_text.insert(tk.END, f"• {key}: {summary[key]}\n")
    else:
        output_text.insert(tk.END, "No objects detected in image.\n")
    output_text.config(state="disabled")

def detect_image():
    output_text.config(state="normal")
    output_text.delete(1.0, tk.END)
    img_label.config(image=None)
    clear_previous_outputs()

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        output_text.config(state="disabled")
        return

    img = Image.open(file_path).convert("RGB")
    temp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(temp_img.name, format="JPEG")
    temp_img.close()

    results = model.predict(
        source=temp_img.name,
        save=True,
        conf=0.25,
        project="runs/detect",
        name="predict",
        exist_ok=True
    )
    save_dir = results[0].save_dir
    detected_images = [f for f in os.listdir(save_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not detected_images:
        output_text.insert(tk.END, "No result image found.\n")
        output_text.config(state="disabled")
        btn_detect.config(text="Select Next Image")
        btn_camera.config(text="Capture Again")
        return

    result_path = os.path.join(save_dir, detected_images[0])
    display_result_image(result_path)
    handle_detection_results(results)
    btn_detect.config(text="Select Next Image")
    btn_camera.config(text="Capture Again")

def capture_from_camera():
    output_text.config(state="normal")
    output_text.delete(1.0, tk.END)
    img_label.config(image=None)
    clear_previous_outputs()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        output_text.insert(tk.END, "Could not open webcam.\n")
        output_text.config(state="disabled")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        output_text.insert(tk.END, "Failed to capture image from camera.\n")
        output_text.config(state="disabled")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    temp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    Image.fromarray(frame_rgb).save(temp_img.name, format="JPEG")
    temp_img.close()

    results = model.predict(
        source=temp_img.name,
        save=True,
        conf=0.25,
        project="runs/detect",
        name="predict",
        exist_ok=True
    )
    save_dir = results[0].save_dir
    detected_images = [f for f in os.listdir(save_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not detected_images:
        output_text.insert(tk.END, "No result image found.\n")
        output_text.config(state="disabled")
        btn_detect.config(text="Select Next Image")
        btn_camera.config(text="Capture Again")
        return

    result_path = os.path.join(save_dir, detected_images[0])
    display_result_image(result_path)
    handle_detection_results(results)
    btn_detect.config(text="Select Next Image")
    btn_camera.config(text="Capture Again")


def live_video_feed():
    global live_running, cap_live
    if not live_running:
        output_text.config(state="normal")
        output_text.delete(1.0, tk.END)
        img_label.config(image=None)
        clear_previous_outputs()
        cap_live = cv2.VideoCapture(0)
        if not cap_live.isOpened():
            output_text.insert(tk.END, "Could not open webcam.\n")
            output_text.config(state="disabled")
            cap_live = None
            return
        live_running = True
        btn_live.config(text="Stop Live Video Feed Detection")
        update_live_feed()
    else:
        stop_live_feed()

def stop_live_feed():
    global live_running, cap_live
    live_running = False
    btn_live.config(text="Live Video Feed Detection")
    if cap_live:
        cap_live.release()
        cap_live = None
    img_label.config(image=None)
    output_text.config(state="normal")
    output_text.delete(1.0, tk.END)
    output_text.config(state="disabled")

def update_live_feed():
    global live_running, cap_live
    if not live_running or cap_live is None:
        return
    ret, frame = cap_live.read()
    if not ret:
        stop_live_feed()
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, conf=0.25)
    annotated = results[0].plot()
    img = Image.fromarray(annotated)
    img = ImageOps.pad(img, (430, 310), color=BG_PANEL)
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk
    handle_detection_results(results)
    img_label.after(30, update_live_feed)

# Control panel at bottom
control_panel = tk.Frame(main_container, bg=BG_MAIN)
control_panel.pack(fill="x", pady=(10, 0))

# Buttons container with better spacing
btn_frame = tk.Frame(control_panel, bg=BG_MAIN)
btn_frame.pack(pady=15)

btn_detect = tk.Button(
    btn_frame,
    text="📁 SELECT IMAGE",
    command=detect_image,
    font=("Montserrat", 11, "bold"),
    bg=ACCENT,
    fg="#0D1117",
    activebackground="#00CC66",
    activeforeground="#F0F6FC",
    bd=0,
    relief=tk.FLAT,
    padx=20, pady=12, cursor="hand2",
    width=15
)
btn_detect.pack(side="left", padx=8)

btn_camera = tk.Button(
    btn_frame,
    text="📷 CAPTURE",
    command=capture_from_camera,
    font=("Montserrat", 11, "bold"),
    bg=ACCENT,
    fg="#0D1117",
    activebackground="#00CC66",
    activeforeground="#F0F6FC",
    bd=0,
    relief=tk.FLAT,
    
    padx=20, pady=12, cursor="hand2",
    width=15
)
btn_camera.pack(side="left", padx=8)

btn_live = tk.Button(
    btn_frame,
    text="🎥 LIVE FEED",
    command=live_video_feed,
    font=("Montserrat", 11, "bold"),
    bg=ACCENT,
    fg="#0D1117",
    activebackground="#00CC66",
    activeforeground="#F0F6FC",
    bd=0,
    relief=tk.FLAT,
    padx=20, pady=12, cursor="hand2",
    width=15
)
btn_live.pack(side="left", padx=8)

def on_close():
    global cap_live, live_running
    stop_live_feed()
    root.destroy()
root.protocol("WM_DELETE_WINDOW", on_close)
# Footer with better positioning
footer = tk.Frame(main_container, bg=BG_MAIN)
footer.pack(side="bottom", fill="x", pady=(15, 0))

# Add separator line
separator = tk.Frame(footer, bg=ACCENT, height=1)
separator.pack(fill="x", pady=(0, 10))

tk.Label(
    footer,
    text="🦺 PPE Compliance Detection System | YOLOv8 Fine-Tuned Model 🦺",
    font=("Montserrat", 9, "italic"),
    fg="#B084CC",
    bg=BG_MAIN
).pack()

root.mainloop()
