import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np
import mediapipe as mp

classes = ['0', '1', '2', '3', '4', '5']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load('../Training/weights/resnet34_hagrid_best.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(1)  # change with 0 if you are using the webcam

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

while True:
    ret, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    total_prediction = 0
    num_hands = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = w
            y_min = h
            x_max = y_max = 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            hand_image = rgb_frame[y_min:y_max, x_min:x_max]
            
            if hand_image.size != 0:
                image = transform(hand_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(image)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    _, predicted = torch.max(output, 1)
                    label = classes[predicted.item()]
                    accuracy = probabilities[predicted.item()].item()
                
                total_prediction += int(label)
                num_hands += 1
            
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                
                text = f"{label}, Acc: {accuracy:.2f}"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_w, text_h = text_size
                cv2.rectangle(frame, (x_min, y_min - text_h - 10), (x_min + text_w, y_min), (255, 0, 0), -1)
                cv2.putText(frame, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if num_hands > 1:
        cv2.putText(frame, f'Total: {total_prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.putText(frame, "Resnet34 (Hagrid)", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()