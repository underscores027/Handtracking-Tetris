import cv2
import mediapipe as mp

# Inicialize o MediaPipe Hands e o módulo de desenho
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inicie a captura de vídeo
cap = cv2.VideoCapture("handtrack.mp4")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Falha ao capturar imagem.")
        break

    # Conversão RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Converta de volta para BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Desenhe landmarks, se detectados
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostre o resultado
    cv2.imshow('Hand Tracking', image)

    # Saia do loop com a tecla 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Libere a câmera e feche as janelas
cap.release()
cv2.destroyAllWindows()
