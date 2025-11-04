import cv2
import torch
import torchvision as tv
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

# ===== CONFIG =====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = './model_Transfer_ep=85_acc=0.8117.pt' 
NUM_CLASSES = 200  # same as training
CLASS_NAMES_PATH = './computer-vision-birds/notebooks/data/CUB_200_2011/classes.txt'  

# ===== LOAD CLASS LABELS =====
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    # class_names.insert(0, "filler")

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

