import cv2
import torch
import torchvision as tv
import torchvision.transforms as transforms
import numpy as np
import os
import smtplib
import time
from datetime import datetime
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional
from PIL import Image

# ==== UTILITIES ====
def load_dotenv_from_cwd(filename=".env") -> None:
    """Simple dotenv loader that leaves existing environment variables untouched."""
    env_path = Path.cwd() / filename
    if not env_path.is_file():
        return

    for line in env_path.read_text().splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        if cleaned.startswith("export "):
            cleaned = cleaned.split(" ", 1)[1]
        if "=" not in cleaned:
            continue
        key, value = cleaned.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# ===== CONFIG =====
load_dotenv_from_cwd()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = './model_Transfer_ep=85_acc=0.8117.pt' 
NUM_CLASSES = 200  # same as training
CLASS_NAMES_PATH = './CUB_200_2011/CUB_200_2011/classes.txt'  
EMAIL_CONFIDENCE_THRESHOLD = float(os.getenv("BIRDSTREAM_EMAIL_CONFIDENCE_THRESHOLD", "75"))
EMAIL_SMTP_SERVER = os.getenv("BIRDSTREAM_EMAIL_SMTP_SERVER")
EMAIL_SMTP_PORT = int(os.getenv("BIRDSTREAM_EMAIL_SMTP_PORT", "587"))
EMAIL_USERNAME = os.getenv("BIRDSTREAM_EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("BIRDSTREAM_EMAIL_PASSWORD")
EMAIL_FROM = os.getenv("BIRDSTREAM_EMAIL_FROM")
EMAIL_TO = os.getenv("BIRDSTREAM_EMAIL_TO")
EMAIL_USE_TLS = os.getenv("BIRDSTREAM_EMAIL_USE_TLS", "1") != "0"
EMAIL_USE_SSL = os.getenv("BIRDSTREAM_EMAIL_USE_SSL", "0") == "1"
EMAIL_ENABLED = os.getenv("BIRDSTREAM_EMAIL_ENABLED", "0") == "1"
EMAIL_TIMEOUT = float(os.getenv("BIRDSTREAM_EMAIL_TIMEOUT", "10"))
EMAIL_ALERT_COOLDOWN = float(os.getenv("BIRDSTREAM_EMAIL_ALERT_COOLDOWN", "300"))
EMAIL_ATTACH_IMAGE = os.getenv("BIRDSTREAM_EMAIL_ATTACH_IMAGE", "1") != "0"
last_alert_times: Dict[str, float] = {}

# Email alert settings rely on environment variables to keep secrets off disk. Enable
# the feature with BIRDSTREAM_EMAIL_ENABLED=1 and configure the SMTP credentials.

def encode_image_bytes(image: Image.Image, fmt="JPEG", quality=85) -> bytes:
    """Encode a PIL image into bytes for attachment."""
    buffer = BytesIO()
    image.save(buffer, format=fmt, quality=quality)
    return buffer.getvalue()


def send_alert_email(
    label: str,
    confidence: float,
    image_bytes: Optional[bytes] = None,
    image_filename: str = "detected_bird.jpg",
) -> bool:
    """Send alert via SMTP when enabled and configured."""
    if not EMAIL_ENABLED:
        return False

    required_fields = (EMAIL_SMTP_SERVER, EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_FROM, EMAIL_TO)
    if not all(required_fields):
        print("Email alert skipped: SMTP configuration incomplete.")
        return False

    msg = EmailMessage()
    msg["Subject"] = f"BirdStream alert: {label} ({confidence:.1f}%)"
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    timestamp = datetime.now().isoformat()
    msg.set_content(
        f"BirdStream detected **{label}** with {confidence:.1f}% confidence at {timestamp}.\n"
        "Message sent automatically by the Raspberry Pi classifier."
    )
    if image_bytes is not None:
        msg.add_attachment(
            image_bytes,
            maintype="image",
            subtype="jpeg",
            filename=image_filename,
        )

    try:
        if EMAIL_USE_SSL:
            server = smtplib.SMTP_SSL(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, timeout=EMAIL_TIMEOUT)
        else:
            server = smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, timeout=EMAIL_TIMEOUT)
            if EMAIL_USE_TLS:
                server.starttls()
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as err:
        print("Email alert failed:", err)
        return False

    print(f"Email alert sent for {label} at {confidence:.1f}%")
    return True

# ===== LOAD CLASS LABELS =====
def load_class_labels(path: str) -> list[str]:
    labels: list[str] = []
    with open(path, 'r') as f:
        for lineno, raw in enumerate(f, 1):
            text = raw.strip()
            if not text:
                continue
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                print(f"Skipping malformed class line {lineno}: {raw!r}")
                continue
            labels.append(parts[1])
    if not labels:
        raise RuntimeError(f"No class labels loaded from {path}")
    return labels

class_names = load_class_labels(CLASS_NAMES_PATH)

print(class_names)
# ===== LOAD MODEL =====
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ===== TRANSFORM (same as test set) =====
def pad(img, fill=(124, 116, 104), size_max=500):
    """Pad a PIL image to a square of size_max."""
    width, height = img.size
    pad_height = max(0, size_max - height)
    pad_width = max(0, size_max - width)
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)


transform = transforms.Compose([
    transforms.Lambda(lambda x: pad(x)),
    transforms.CenterCrop((375, 375)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== WEBCAM LOOP =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

frameCount = 0
top_label= ""
top_conf = 0
lbl = ""
conf = 0


print("Press 'q' to quit")
with torch.no_grad():

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frameCount += 1
        # Convert BGR → RGB PIL
        from PIL import Image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Apply padding + center crop manually
        padded = pad(pil_img)
        cropped = transforms.CenterCrop((375, 375))(padded)
        display_img = np.array(cropped)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

        if (frameCount == 60):
            img_tensor = transforms.ToTensor()(cropped)
            img_tensor = transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])(img_tensor)
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)


            # Model inference
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top5_prob, top5_catid = torch.topk(probs, 5)

            # Overlay top-1 prediction
            top_label = class_names[top5_catid[0, 0]]
            top_conf  = top5_prob[0, 0].item() * 100
            now = time.time()

            if top_conf >= EMAIL_CONFIDENCE_THRESHOLD:
                last_sent = last_alert_times.get(top_label)
                cooldown_elapsed = last_sent is None or (now - last_sent) >= EMAIL_ALERT_COOLDOWN
                if cooldown_elapsed:
                    image_bytes = None
                    image_filename = "detected_bird.jpg"
                    if EMAIL_ATTACH_IMAGE:
                        safe_label = top_label.replace(" ", "_")
                        image_filename = f"{safe_label}_{int(now)}.jpg"
                        image_bytes = encode_image_bytes(cropped)
                    if send_alert_email(top_label, top_conf, image_bytes, image_filename):
                        last_alert_times[top_label] = now

            # Optionally overlay smaller top-5 list
            for i in range(top5_prob.size(1)):
                lbl = class_names[top5_catid[0, i]]
                conf = top5_prob[0, i].item() * 100

            frameCount = 0
        cv2.putText(display_img, f"{top_label} ({top_conf:.1f}%)", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(display_img, f"{lbl}: {conf:.1f}%", (15, 70 + 30*i),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        # Show exactly the cropped image
        cv2.imshow("Bird Classifier (cropped view)", display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
