import cv2
import mediapipe as mp
import numpy as np
import random
import time
from collections import deque

# Kamera başlatılıyor
cap = cv2.VideoCapture(0)

# Yüz algılama modeli oluşturuluyor
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# El algılama modeli oluşturuluyor
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Yapboz ayarları
GRID = 3
PUZZLE_SIZE = 360
WINDOW_NAME = "Iki Kisilik Yuz Yapboz"
GRAB_FINGER_COUNT = 3

# Oyuncuların yüzleri burada saklanır
player_faces = [None, None]
current_face_crop = None

# Yapboz parçaları ve sıralama bilgileri
pieces = []
slots = []
start_slots = []

# Seçilen parçanın bilgileri
selected_piece = None
selected_from_slot = None
selected_pos = [0, 0]
target_pos = [0, 0]

# Oyunun hangi ekranda olduğunu tutar
stage = "menu"
turn_start_time = None
countdown_start_time = None
countdown_next_stage = None

# Kazanma animasyonu bilgileri
win_anim_start = None
win_next_stage = None
win_player = None
win_time = None

# Oyuncuların süreleri
p1_time = None
p2_time = None

# El izi ve konfeti efektleri için listeler
trail_points = deque(maxlen=25)
confetti = []


# Verilen yüz görüntüsünden yapboz parçaları oluşturulur
def make_puzzle(img):
    global pieces, slots, start_slots

    pieces = []
    piece_size = PUZZLE_SIZE // GRID

    for y in range(GRID):
        for x in range(GRID):
            piece = img[
                y * piece_size:(y + 1) * piece_size,
                x * piece_size:(x + 1) * piece_size
            ]
            pieces.append(piece)

    slots = list(range(GRID * GRID))

    # Hiçbir parça doğru yerde başlamasın diye karıştırma kontrol edilir
    while True:
        random.shuffle(slots)
        if all(slots[i] != i for i in range(GRID * GRID)):
            break

    start_slots = slots.copy()


# Mevcut yapbozu ilk karışık haline döndürür
def reset_current_puzzle():
    global slots, selected_piece, selected_from_slot
    slots = start_slots.copy()
    selected_piece = None
    selected_from_slot = None


# Yapbozun çözülüp çözülmediğini kontrol eder
def is_solved():
    return slots == list(range(GRID * GRID))


# Elin avuç/yumruk şeklinde kapalı olup olmadığını kontrol eder
def is_hand_grabbing(hand_landmarks):
    lm = hand_landmarks.landmark
    folded_count = 0

    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        if lm[tip].y > lm[pip].y:
            folded_count += 1

    return folded_count >= GRAB_FINGER_COUNT


# Elin hangi yapboz karesinin üstünde olduğunu bulur
def get_slot_at(x, y):
    piece_size = PUZZLE_SIZE // GRID
    grid_x = (x - 100) // piece_size
    grid_y = (y - 80) // piece_size

    if 0 <= grid_x < GRID and 0 <= grid_y < GRID:
        return int(grid_y * GRID + grid_x)

    return None


# Ekrana normal yazı yazar
def draw_text(img, text, x, y, size=1, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick)


# Ekrana ortalanmış yazı yazar
def draw_center_text(img, text, y, size=1.2, color=(255, 255, 255), thick=2):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thick)[0]
    x = (img.shape[1] - text_size[0]) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick)


# Menü ve bilgi ekranlarında kullanılan paneli çizer
def draw_panel(img, x1, y1, x2, y2):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (25, 25, 35), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2)


# El göstergesi için parlama efekti çizer
def draw_glow_circle(img, center, radius, color):
    x, y = center
    for r in range(radius + 18, radius, -6):
        cv2.circle(img, (x, y), r, color, 1)
    cv2.circle(img, (x, y), radius, color, -1)


# Seçilen parça veya yüz için parlama efekti çizer
def draw_glow_rect(img, x, y, w, h, color):
    for t in range(10, 0, -3):
        cv2.rectangle(img, (x - t, y - t), (x + w + t, y + h + t), color, 1)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)


# Elin hareket ettiği yerlerde iz efekti oluşturur
def draw_trail(display):
    for i, point in enumerate(trail_points):
        if point is None:
            continue

        x, y = point
        radius = int(4 + (i / len(trail_points)) * 14)
        brightness = int(80 + (i / len(trail_points)) * 175)
        cv2.circle(display, (x, y), radius, (0, brightness, brightness), -1)


# Kazanma ekranı için konfeti parçaları oluşturur
def create_confetti():
    global confetti

    confetti = []

    for _ in range(80):
        confetti.append([
            random.randint(0, 700),
            random.randint(-500, 0),
            random.randint(3, 8),
            random.randint(2, 7),
            random.choice([
                (0, 255, 255),
                (0, 255, 0),
                (255, 0, 255),
                (255, 255, 255),
                (255, 120, 0)
            ])
        ])


# Konfeti efektini ekrana çizer
def draw_confetti(display):
    for c in confetti:
        c[1] += c[3]

        if c[1] > 600:
            c[0] = random.randint(0, 700)
            c[1] = random.randint(-100, 0)

        x, y, size, speed, color = c
        cv2.rectangle(display, (x, y), (x + size, y + size), color, -1)


# Kameradan yüzü algılar ve yüz bölgesini keser
def detect_face(frame):
    global current_face_crop

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    current_face_crop = None

    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        x = max(0, x - 50)
        y = max(0, y - 90)
        bw = min(w - x, bw + 100)
        bh = min(h - y, bh + 130)

        if bw > 0 and bh > 0:
            current_face_crop = frame[y:y + bh, x:x + bw]
            return x, y, bw, bh

    return None


# Oyun başlamadan önce geri sayım ekranına geçer
def start_countdown(next_stage):
    global stage, countdown_start_time, countdown_next_stage

    stage = "countdown"
    countdown_start_time = time.time()
    countdown_next_stage = next_stage


# Oyuncu turu bitince kazanma animasyonunu başlatır
def start_win_animation(player, finish_time, next_stage):
    global stage, win_anim_start, win_next_stage, win_player, win_time

    stage = "win_anim"
    win_anim_start = time.time()
    win_next_stage = next_stage
    win_player = player
    win_time = finish_time
    create_confetti()


# Pencere tam ekran yapılır
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# Ana oyun döngüsü
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    # Başlangıç menüsü
    if stage == "menu":
        display = np.zeros((600, 700, 3), dtype=np.uint8)

        draw_panel(display, 70, 80, 630, 500)
        draw_center_text(display, "2 KISILIK", 170, 1.4, (0, 255, 255), 3)
        draw_center_text(display, "YUZ YAPBOZ OYUNU", 230, 1.2, (255, 255, 255), 2)
        draw_center_text(display, "Kamera + Yuz Tespiti + El Kontrolu", 300, 0.7, (180, 180, 180), 2)
        draw_center_text(display, "ENTER = Basla", 380, 0.9, (0, 255, 0), 2)
        draw_center_text(display, "ESC = Cikis", 430, 0.7, (200, 200, 200), 2)

        if key == 13:
            stage = "capture_p1"

        cv2.imshow(WINDOW_NAME, display)
        continue

    # Oyuncuların yüzlerini kaydetme bölümü
    if stage in ["capture_p1", "capture_p2"]:
        display = frame.copy()
        face_box = detect_face(frame)
        player_no = 1 if stage == "capture_p1" else 2

        draw_text(display, f"Oyuncu {player_no} yuzunu kameraya goster", 40, 60, 1, (0, 255, 255), 2)

        if face_box is not None:
            x, y, bw, bh = face_box
            draw_glow_rect(display, x, y, bw, bh, (0, 255, 0))

        if key == ord("c") and current_face_crop is not None:
            saved_face = cv2.resize(current_face_crop, (PUZZLE_SIZE, PUZZLE_SIZE))

            if stage == "capture_p1":
                player_faces[0] = saved_face
                stage = "capture_p2"
            else:
                player_faces[1] = saved_face
                stage = "ready_p1"

        cv2.imshow(WINDOW_NAME, display)
        continue

    # Oyuncu 1 başlamadan önceki bekleme ekranı
    if stage == "ready_p1":
        display = np.zeros((600, 700, 3), dtype=np.uint8)

        draw_panel(display, 60, 120, 640, 470)
        draw_center_text(display, "YUZLER KAYDEDILDI", 190, 1, (0, 255, 0), 2)
        draw_center_text(display, "Oyuncu 1 hazirlan", 255, 1, (255, 255, 255), 2)
        draw_center_text(display, "N = Oyunu baslat", 390, 0.85, (255, 255, 255), 2)

        if key == ord("n"):
            make_puzzle(player_faces[1])
            trail_points.clear()
            start_countdown("play_p1")

        cv2.imshow(WINDOW_NAME, display)
        continue

    # Oyuncu 2 başlamadan önceki bekleme ekranı
    if stage == "wait_p2":
        display = np.zeros((600, 700, 3), dtype=np.uint8)

        draw_panel(display, 60, 120, 640, 480)
        draw_center_text(display, f"Oyuncu 1 bitirdi: {p1_time:.2f} sn", 205, 0.9, (0, 255, 0), 2)
        draw_center_text(display, "Oyuncu 2 hazirlan", 270, 1, (255, 255, 255), 2)
        draw_center_text(display, "N = Oyunu baslat", 405, 0.85, (255, 255, 255), 2)

        if key == ord("n"):
            make_puzzle(player_faces[0])
            trail_points.clear()
            start_countdown("play_p2")

        cv2.imshow(WINDOW_NAME, display)
        continue

    # 3-2-1 geri sayım ekranı
    if stage == "countdown":
        display = np.zeros((600, 700, 3), dtype=np.uint8)

        elapsed = time.time() - countdown_start_time
        remaining = 3 - int(elapsed)

        if elapsed < 3:
            size = 4 + (elapsed % 1) * 1.2
            draw_center_text(display, str(remaining), 330, size, (0, 255, 255), 6)
        elif elapsed < 4:
            draw_center_text(display, "BASLA!", 330, 2.2, (0, 255, 0), 5)
        else:
            stage = countdown_next_stage
            turn_start_time = time.time()

        cv2.imshow(WINDOW_NAME, display)
        continue

    # Tur bitince gösterilen kısa animasyon
    if stage == "win_anim":
        display = np.zeros((600, 700, 3), dtype=np.uint8)

        draw_confetti(display)

        elapsed = time.time() - win_anim_start
        pulse = 1 + 0.15 * np.sin(elapsed * 10)

        draw_center_text(display, f"OYUNCU {win_player} BITIRDI!", 250, 1.4 * pulse, (0, 255, 0), 4)
        draw_center_text(display, f"Sure: {win_time:.2f} sn", 330, 1.0, (0, 255, 255), 2)

        if win_next_stage != "result":
            draw_center_text(display, "Devam ediliyor...", 430, 0.7, (220, 220, 220), 2)

        if elapsed >= 2.5:
            stage = win_next_stage

        cv2.imshow(WINDOW_NAME, display)
        continue

    # Oyunun sonuç ekranı
    if stage == "result":
        display = np.zeros((600, 700, 3), dtype=np.uint8)

        draw_confetti(display)
        draw_panel(display, 70, 90, 630, 500)
        draw_center_text(display, "SONUC", 155, 1.4, (255, 255, 255), 3)
        draw_center_text(display, f"Oyuncu 1: {p1_time:.2f} sn", 250, 1, (0, 255, 255), 2)
        draw_center_text(display, f"Oyuncu 2: {p2_time:.2f} sn", 305, 1, (0, 255, 255), 2)

        if p1_time < p2_time:
            winner = "Oyuncu 1 kazandi!"
        elif p2_time < p1_time:
            winner = "Oyuncu 2 kazandi!"
        else:
            winner = "Berabere!"

        draw_center_text(display, winner, 395, 1, (0, 255, 0), 3)
        draw_center_text(display, "ESC = cikis", 455, 0.7, (220, 220, 220), 2)

        cv2.imshow(WINDOW_NAME, display)
        continue

    # Asıl yapboz oyun ekranı
    if stage in ["play_p1", "play_p2"]:
        display = np.zeros((600, 700, 3), dtype=np.uint8)
        piece_size = PUZZLE_SIZE // GRID

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)

        grab = False
        hand_x = None
        hand_y = None

        # El algılanırsa elin merkezi ve tutma durumu hesaplanır
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark

            hand_x = int(lm[9].x * 700)
            hand_y = int(lm[9].y * 600)

            trail_points.append((hand_x, hand_y))
            grab = is_hand_grabbing(hand_landmarks)
        else:
            trail_points.append(None)

        current_player = 1 if stage == "play_p1" else 2
        target_player = 2 if stage == "play_p1" else 1
        elapsed_time = time.time() - turn_start_time

        draw_text(display, f"Oyuncu {current_player} oynuyor", 30, 35, 0.8, (255, 255, 255), 2)
        draw_text(display, f"Hedef: Oyuncu {target_player} yuzu", 30, 65, 0.7, (0, 255, 255), 2)
        draw_text(display, f"Sure: {elapsed_time:.1f} sn", 500, 35, 0.8, (255, 255, 255), 2)
        draw_text(display, "R = turu basa al, ESC = cikis", 30, 585, 0.6, (180, 180, 180), 1)

        if key == ord("r"):
            reset_current_puzzle()

        # El kapalıysa parça tutulur veya sürüklenir
        if grab and hand_x is not None and hand_y is not None:
            if selected_piece is None:
                slot_index = get_slot_at(hand_x, hand_y)

                if slot_index is not None:
                    selected_piece = slots[slot_index]
                    selected_from_slot = slot_index

                    selected_pos = [
                        100 + (slot_index % GRID) * piece_size,
                        80 + (slot_index // GRID) * piece_size
                    ]

                    target_pos = selected_pos.copy()
            else:
                target_pos[0] = hand_x - piece_size // 2
                target_pos[1] = hand_y - piece_size // 2

                # Parça elin arkasından yumuşak şekilde hareket eder
                selected_pos[0] += int((target_pos[0] - selected_pos[0]) * 0.35)
                selected_pos[1] += int((target_pos[1] - selected_pos[1]) * 0.35)

        # El açılınca parça bırakılır ve gerekirse diğer parça ile yer değiştirir
        else:
            if selected_piece is not None:
                center_x = selected_pos[0] + piece_size // 2
                center_y = selected_pos[1] + piece_size // 2
                drop_slot = get_slot_at(center_x, center_y)

                if drop_slot is not None:
                    old_piece = slots[drop_slot]
                    slots[drop_slot] = selected_piece
                    slots[selected_from_slot] = old_piece
                else:
                    slots[selected_from_slot] = selected_piece

                selected_piece = None
                selected_from_slot = None

        # Yapboz parçaları ekrana çizilir
        for i in range(GRID * GRID):
            x = 100 + (i % GRID) * piece_size
            y = 80 + (i // GRID) * piece_size

            if i == selected_from_slot and selected_piece is not None:
                cv2.rectangle(display, (x, y), (x + piece_size, y + piece_size), (50, 50, 50), 2)
                continue

            piece_index = slots[i]
            display[y:y + piece_size, x:x + piece_size] = pieces[piece_index]
            cv2.rectangle(display, (x, y), (x + piece_size, y + piece_size), (255, 255, 255), 2)

        # Tutulan parça en üstte gösterilir
        if selected_piece is not None:
            px, py = selected_pos

            if 0 <= px <= 700 - piece_size and 0 <= py <= 600 - piece_size:
                display[py:py + piece_size, px:px + piece_size] = pieces[selected_piece]

            draw_glow_rect(display, px, py, piece_size, piece_size, (0, 255, 255))

        draw_trail(display)

        # El göstergesi çizilir
        if hand_x is not None and hand_y is not None:
            color = (0, 255, 0) if grab else (0, 0, 255)
            draw_glow_circle(display, (hand_x, hand_y), 16, color)

        if grab:
            draw_text(display, "TUTUYOR", 500, 585, 0.7, (0, 255, 0), 2)
        else:
            draw_text(display, "BIRAKTI", 500, 585, 0.7, (0, 0, 255), 2)

        # Yapboz tamamlanınca ilgili oyuncunun süresi kaydedilir
        if is_solved():
            if stage == "play_p1":
                p1_time = elapsed_time
                selected_piece = None
                selected_from_slot = None
                start_win_animation(1, p1_time, "wait_p2")
            else:
                p2_time = elapsed_time
                selected_piece = None
                selected_from_slot = None
                start_win_animation(2, p2_time, "result")

        cv2.imshow(WINDOW_NAME, display)

# Kamera ve pencereler kapatılır
cap.release()
cv2.destroyAllWindows() 