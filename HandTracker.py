import cv2
import mediapipe as mp
import time
import pyautogui

class HandDetector:
    def __init__(self, static_mode=False, max_hands=2, detection_confidence=0.7, track_confidence=0.5):
        self.mode = static_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        
        # Inicializa o MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.track_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, image, draw=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)
        
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return image
    
    def find_position(self, image, hand_no=0, draw=True):
        lm_list = []
        
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            
            for id, lm in enumerate(hand.landmark):
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                
                if draw:
                    cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        
        return lm_list

    def is_index_finger_open(self, lm_list):
        if lm_list:
            return lm_list[self.tip_ids[1]][2] < lm_list[self.tip_ids[1] - 2][2]
        return False

    def is_thumb_up(self, lm_list):
        if lm_list:
            return lm_list[self.tip_ids[0]][1] > lm_list[self.tip_ids[0] - 1][1]
        return False

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    last_check_time = time.time()
    
    left_thumb_up_prev, right_thumb_up_prev = False, False
    left_index_open_prev, right_index_open_prev = False, False
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Falha ao capturar imagem.")
            break
        
        image = detector.find_hands(image)
        
        # Inicializa variáveis de estado dos dedos
        left_index_open = False
        right_index_open = False
        left_thumb_up = False
        right_thumb_up = False

        if detector.results.multi_hand_landmarks:
            for hand_no, hand_landmarks in enumerate(detector.results.multi_hand_landmarks):
                lm_list = detector.find_position(image, hand_no)
                
                if hand_no == 0:
                    left_index_open = detector.is_index_finger_open(lm_list)
                    left_thumb_up = detector.is_thumb_up(lm_list)
                elif hand_no == 1:
                    right_index_open = detector.is_index_finger_open(lm_list)
                    right_thumb_up = detector.is_thumb_up(lm_list)

        # Verifica se já passou 1 segundo desde a última detecção
        if time.time() - last_check_time > 1:
            # Rotaciona a peça (W) - ambos os polegares mudam de aberto para fechado
            if (left_thumb_up_prev and right_thumb_up_prev) and not (left_thumb_up or right_thumb_up):
                pyautogui.press('w')

            # Move a peça para a esquerda (A) - mudança do polegar esquerdo de aberto para fechado
            elif left_thumb_up_prev and not left_thumb_up:
                pyautogui.press('a')

            # Move a peça para a direita (D) - mudança do polegar direito de aberto para fechado
            elif right_thumb_up_prev and not right_thumb_up:
                pyautogui.press('d')

            # Aumenta a velocidade de queda da peça (S) - mudança de qualquer indicador de aberto para fechado
            elif (left_index_open_prev and not left_index_open) or (right_index_open_prev and not right_index_open):
                pyautogui.press('s')

            # Derruba a peça direto (Barra de espaço) - todos os indicadores e polegares mudam de aberto para fechado
            if (left_thumb_up_prev and right_thumb_up_prev and left_index_open_prev and right_index_open_prev and
                not (left_thumb_up or right_thumb_up or left_index_open or right_index_open)):
                pyautogui.press('space')

            # Atualiza os estados dos dedos
            left_thumb_up_prev, right_thumb_up_prev = left_thumb_up, right_thumb_up
            left_index_open_prev, right_index_open_prev = left_index_open, right_index_open

            last_check_time = time.time()  # Reseta o temporizador para verificar novamente em 1 segundo

        # Exibe os estados na imagem
        cv2.putText(image, f'Polegar Esq: {"Aberto" if left_thumb_up else "Fechado"}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f'Polegar Dir: {"Aberto" if right_thumb_up else "Fechado"}', (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(image, f'Indicador Esq: {"Aberto" if left_index_open else "Fechado"}', (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f'Indicador Dir: {"Aberto" if right_index_open else "Fechado"}', (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Exibe a imagem com os estados
        cv2.imshow('Hand Tracking', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
