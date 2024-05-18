# import cv2
# import mediapipe as mp
# import time


# data = []
# labels = []

# class HandDetector():
#     def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionConfidence=0.5, trackConfidence=0.5):
#         self.mode = mode
#         self.maxHands = maxHands
#         self.modelComplexity = modelComplexity
#         self.detectionConfidence = detectionConfidence
#         self.trackConfidence = trackConfidence

#         self.mpHands = mp.solutions.hands
#         self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConfidence, self.trackConfidence)
#         self.mpDraw = mp.solutions.drawing_utils


#     # def findHands(self, img, draw=True):
#     #     img = cv2.flip(img, 1)
#     #     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     #     self.results = self.hands.process(imgRGB)

#     #     if self.results.multi_hand_landmarks:
#     #         for handLms in self.results.multi_hand_landmarks:
#     #             if draw:
#     #                 self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
#     #     return img
#     def findHands(self, img, draw=True):
#         data_aux = []
#         img = cv2.flip(img, 1)
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.hands.process(imgRGB)

#         if self.results.multi_hand_landmarks:
#             for handLms in self.results.multi_hand_landmarks:
#                 if draw:
#                     self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
#                 for i in range(len(handLms.landmark)):
#                     x = handLms.landmark[i].x
#                     y = handLms.landmark[i].y        
#                    # print(f"Landmark {i}: x={x}, y={y}")
#                     data_aux.append(x)
#                     data_aux.append(y)
#             data.append(data_aux)
#             # append labels from the directory
#         return img

                
        

#     def findPosition(self, img, handNo=0, draw=True):
#         lmList = []
#         if self.results.multi_hand_landmarks:
#             myHand = self.results.multi_hand_landmarks[handNo]

#             for id, lm in enumerate(myHand.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lmList.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
#         return lmList


# def main():
#     cap = cv2.VideoCapture(1)  # 0 for webcam, 1 for external webcam
#     detector = HandDetector()

#     while True:
#         success, img = cap.read()
#         if not success:
#             print("Impossibile leggere il frame dalla webcam.")
#             break
        
#         img = detector.findHands(img)
#         lmList = detector.findPosition(img)
        
#         cv2.imshow("Image", img)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



import cv2
import mediapipe as mp
import torch
from model import HandModel  
import numpy as np

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

        self.hand_classifier = HandModel(400 * 400 * 3, 128, number_of_classes)  
        self.hand_classifier.load_state_dict(torch.load("weights/HandModel-50.pth"))  
        self.hand_classifier.eval()  

    def findHands(self, img, draw=True):
        data_aux = []
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                h, w, _ = img.shape
                x_min, x_max = w, 0
                y_min, y_max = h, 0
                for lm in handLms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    x_max = max(x_max, x)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y)
                hand_img = img[y_min:y_max, x_min:x_max]
                hand_img = cv2.resize(hand_img, (400, 400))
                hand_img = np.transpose(hand_img, (2, 0, 1))
                hand_img = torch.tensor(hand_img, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    outputs = self.hand_classifier(hand_img)
                    predicted_class = torch.argmax(outputs, dim=1).item()
                   #print("Predicted class:", predicted_class)
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(img, str(predicted_class), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

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
    cap = cv2.VideoCapture(0)  # 0 for webcam, 1 for external webcam
    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Impossibile leggere il frame dalla webcavoglia comunqueam.")
            break
        
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
