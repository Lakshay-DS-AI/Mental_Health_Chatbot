# hs_module.py
# Modular version of HSVER2.py with functions for data collection, training,
# prediction-from-frame, delete, and show_trained_words.
#
# Edit BASE_DIR to match your dataset/model location if needed.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
import time
from sklearn.ensemble import RandomForestClassifier

# ---------------- SETTINGS ----------------
BASE_DIR = r"D:\New folder\MentalHealthChatbot\M MAIN\Working model"
DATA_FILE = os.path.join(BASE_DIR, "word_data.csv")
REF_DIR = os.path.join(BASE_DIR, "reference_images")
MODEL_FILE = os.path.join(BASE_DIR, "word_model.pkl")

NUM_SAMPLES = 100
REF_SIZE = 400
FEATURES_PER_HAND = 21 * 3  # 21 landmarks × 3
TOTAL_FEATURES = FEATURES_PER_HAND * 2
# ------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ---------------- Helpers ----------------
def _ensure_dirs():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    if not os.path.exists(REF_DIR):
        os.makedirs(REF_DIR)


def _frame_to_features(frame):
    """
    Given a BGR frame (numpy), detect hands and return flattened feature vector:
    63 coords for each hand (x,y,z). If second hand missing, pad with zeros.
    Returns None if no hands detected.
    """
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None
        data_row = []
        # first hand
        hand1 = results.multi_hand_landmarks[0]
        for lm in hand1.landmark:
            data_row.extend([lm.x, lm.y, lm.z])
        # second hand if exists
        if len(results.multi_hand_landmarks) > 1:
            hand2 = results.multi_hand_landmarks[1]
            for lm in hand2.landmark:
                data_row.extend([lm.x, lm.y, lm.z])
        else:
            data_row.extend([0] * FEATURES_PER_HAND)
        return data_row


# ---------------- MODE: Collect Data ----------------
def collect_data(num_samples: int = NUM_SAMPLES):
    """
    Interactive data capture for a word: captures NUM_SAMPLES feature rows
    and saves into DATA_FILE, then optionally captures a reference photo.
    """
    _ensure_dirs()
    word = input("Enter the word to collect (A-Z only): ").strip().upper()
    if not word.isalpha():
        print("Invalid word! Only alphabets allowed.")
        return

    n_samples = num_samples
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
        collected = 0
        last_time = 0
        cooldown_time = 0.3

        print(f"Collecting {n_samples} samples for '{word}'... (Press ESC to abort)")
        while collected < n_samples:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                current_time = time.time()
                if (current_time - last_time) < cooldown_time:
                    cv2.imshow("Data Collection", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue
                last_time = current_time

                data_row = []
                hand1 = results.multi_hand_landmarks[0]
                for lm in hand1.landmark:
                    data_row.extend([lm.x, lm.y, lm.z])
                if len(results.multi_hand_landmarks) > 1:
                    hand2 = results.multi_hand_landmarks[1]
                    for lm in hand2.landmark:
                        data_row.extend([lm.x, lm.y, lm.z])
                else:
                    data_row.extend([0] * FEATURES_PER_HAND)

                data_row.append(word)

                # Save to CSV (append)
                if not os.path.exists(DATA_FILE):
                    df = pd.DataFrame([data_row])
                    df.to_csv(DATA_FILE, index=False, header=False)
                else:
                    df_existing = pd.read_csv(DATA_FILE, header=None)
                    df_new = pd.DataFrame([data_row])
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(DATA_FILE, index=False, header=False)

                collected += 1

                # draw landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.putText(frame, f"Collected {collected}/{n_samples} for '{word}'",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection for '{word}' completed!")

    # reference photo
    print("Press ENTER to capture a reference photo (or type 'skip' + Enter to skip).")
    ans = input().strip().lower()
    if ans == 'skip':
        print("Skipped reference photo.")
        return

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        crop = frame[h // 2 - REF_SIZE // 2:h // 2 + REF_SIZE // 2, w // 2 - REF_SIZE // 2:w // 2 + REF_SIZE // 2]
        if crop.shape[0] != REF_SIZE or crop.shape[1] != REF_SIZE:
            # fallback: resize for safety
            crop = cv2.resize(frame, (REF_SIZE, REF_SIZE))
        cv2.putText(crop, f"Press 's' to save reference for '{word}' (ESC to skip)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Reference Photo", crop)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            ref_path = os.path.join(REF_DIR, f"{word}.jpg")
            cv2.imwrite(ref_path, crop)
            print(f"Saved reference photo: {ref_path}")
            break
        elif key == 27:
            print("Skipped reference photo.")
            break
    cap.release()
    cv2.destroyAllWindows()


# ---------------- MODE: Train Model ----------------
def train_model():
    """
    Train a RandomForestClassifier on DATA_FILE and write MODEL_FILE.
    """
    if not os.path.exists(DATA_FILE):
        print("No data found! Collect data first.")
        return

    df = pd.read_csv(DATA_FILE, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    print("Model trained!")

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved as '{MODEL_FILE}'")


# ---------------- MODE: Predict From Frame ----------------
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE, "rb") as f:
        clf = pickle.load(f)
    return clf


def predict_from_frame(frame, clf=None):
    """
    Given a BGR frame (numpy) and an optional loaded classifier,
    returns predicted word (string) or None if no hands or no model.
    """
    if clf is None:
        clf = load_model()
    if clf is None:
        return None

    features = _frame_to_features(frame)
    if features is None:
        return None
    try:
        pred = clf.predict([features])[0]
        return str(pred)
    except Exception as e:
        print("Prediction error:", e)
        return None


# ---------------- MODE: Delete Word ----------------
def delete_word(word: str):
    """
    Delete all entries of 'word' from the CSV and its reference image.
    Retrains model automatically if data remains.
    """
    if not os.path.exists(DATA_FILE):
        print("No data found! CSV does not exist.")
        return

    word = word.strip().upper()
    if not word.isalpha():
        print("Invalid word! Only alphabets allowed.")
        return

    df = pd.read_csv(DATA_FILE, header=None)
    initial_count = len(df)
    df = df[df.iloc[:, -1].str.upper() != word]  # remove rows matching word
    final_count = len(df)

    if final_count < initial_count:
        df.to_csv(DATA_FILE, index=False, header=False)
        print(f"Deleted all entries of '{word}' from CSV.")
    else:
        print(f"No entries found for '{word}' in CSV.")

    ref_path = os.path.join(REF_DIR, f"{word}.jpg")
    if os.path.exists(ref_path):
        os.remove(ref_path)
        print(f"Deleted reference photo: {ref_path}")
    else:
        print(f"No reference photo found for '{word}'.")

    # Retrain or delete model
    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
        print("Retraining model after deletion...")
        train_model()
    else:
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        print("No data left in CSV. Model deleted.")


# ---------------- MODE: Show Trained Words ----------------
def show_trained_words():
    clf = load_model()
    if clf is None:
        print("No trained model found! Train data first.")
        return
    if hasattr(clf, "classes_"):
        print("\n✅ The model has been trained on the following words:")
        for word in clf.classes_:
            print(" -", word)
    else:
        print("⚠️ This model does not expose classes_ attribute.")


# ---------------- CLI Helper ----------------
def run_cli():
    """
    Keep the old CLI behavior for training & collection in terminal.
    """
    while True:
        print("\nHS Module — Choose mode:")
        print("1: Collect Data")
        print("2: Train Model")
        print("3: Live Prediction (OpenCV window)")
        print("4: Delete Word")
        print("5: Show Trained Words")
        print("0: Exit")
        mode = input("Enter mode: ").strip()
        if mode == "1":
            collect_data()
        elif mode == "2":
            train_model()
        elif mode == "3":
            live_prediction()
        elif mode == "4":
            w = input("Enter word to delete: ").strip()
            delete_word(w)
        elif mode == "5":
            show_trained_words()
        elif mode == "0":
            break
        else:
            print("Invalid mode!")

# keep original live_prediction as interactive OpenCV window (for debug)
def live_prediction():
    """
    Run the interactive live_prediction OpenCV window (original behavior).
    This is kept for CLI usability.
    """
    clf = load_model()
    if clf is None:
        print("No trained model found! Train data first.")
        return

    cap = cv2.VideoCapture(0)
    sentence = ""
    last_pred = None
    pred_frame_count = 0
    stable_frames_required = 5
    no_hand_frames = 0
    no_hand_threshold = 10
    capitalize_next = True
    used_words = set()
    key_delay = 0.3
    key_cooldowns = {"space": 0, "backspace": 0, ".": 0, ",": 0, "?": 0, "!": 0}

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            current_time = time.time()

            if results.multi_hand_landmarks:
                no_hand_frames = 0
                data_row = []
                hand1 = results.multi_hand_landmarks[0]
                for lm in hand1.landmark:
                    data_row.extend([lm.x, lm.y, lm.z])
                if len(results.multi_hand_landmarks) > 1:
                    hand2 = results.multi_hand_landmarks[1]
                    for lm in hand2.landmark:
                        data_row.extend([lm.x, lm.y, lm.z])
                else:
                    data_row.extend([0] * FEATURES_PER_HAND)

                pred_word = clf.predict([data_row])[0]

                if pred_word == last_pred:
                    pred_frame_count += 1
                else:
                    pred_frame_count = 1
                    last_pred = pred_word

                if pred_frame_count >= stable_frames_required:
                    if pred_word not in [".", ",", "?", "!"]:
                        if pred_word.lower() not in used_words:
                            formatted_word = pred_word.capitalize() if capitalize_next else pred_word.lower()
                            sentence += formatted_word + " "
                            used_words.add(pred_word.lower())
                            capitalize_next = False
                    else:
                        sentence = sentence.strip() + pred_word + " "
                        used_words.clear()
                        capitalize_next = True

                    pred_frame_count = 0
                    last_pred = None

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                no_hand_frames += 1
                if no_hand_frames >= no_hand_threshold:
                    last_pred = None
                    pred_frame_count = 0

            # keyboard handling intentionally omitted for simplicity in modular CLI

            cv2.putText(frame, f"Sentence: {sentence}", (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Live Prediction", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()


# If run directly, run the CLI
if __name__ == "__main__":
    run_cli()
