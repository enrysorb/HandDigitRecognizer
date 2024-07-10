import cv2
import mediapipe as mp
import torch
from model.MLP_CSV import HandModel  
import numpy as np
import joblib

data = []
labels = []
number_of_classes = 6

class HandDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

        self.hand_classifier = HandModel()
        device = torch.device('cpu') 
        self.hand_classifier.load_state_dict(torch.load("./Training/weights/model_CSV.pth", map_location=device))  
        self.hand_classifier.eval()  

    def findHands(self, img, draw=True):
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        predicted_classes = []

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                
                h, w, _ = img.shape
                landmarks = []
                x_min, x_max = w, 0
                y_min, y_max = h, 0
                for lm in handLms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append(x)
                    landmarks.append(y)
                    x_min = min(x_min, x)
                    x_max = max(x_max, x)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y)

                landmarks = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = self.hand_classifier(landmarks)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item() * 100  
                    predicted_classes.append(predicted_class)
                    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}%")
                    
                    label = f"{predicted_class}: {confidence:.2f}%"
                    
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_x, label_y = x_min, y_min - 10
                    cv2.rectangle(img, (label_x, label_y - label_size[1] - 10), (label_x + label_size[0], label_y + 5), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, label, (label_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            if len(predicted_classes) > 1:
                total_class_value = sum(predicted_classes)
                print(f"Total class value (sum of predicted classes): {total_class_value}")
                cv2.putText(img, f"Total: {total_class_value}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture(1)  
    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Impossibile leggere il frame dalla webcam.")
            break
        
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        cv2.putText(img, "MLP", (img.shape[1] - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
