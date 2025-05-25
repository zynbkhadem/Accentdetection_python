import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext



def extract_features(file_path, n_mfcc=40):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        return np.concatenate((mfcc_mean, mfcc_std, mfcc_delta_mean))
    except Exception as e:
        return None

def prepare_dataset(dataset_path):
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5)
    ])
    features, labels = [], []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.endswith(('.wav', '.mp3')):
                    path = os.path.join(label_path, file)
                    audio, sr = librosa.load(path, sr=16000)
                    feat = extract_features(path)
                    if feat is not None:
                        features.append(feat)
                        labels.append(label)
                    for _ in range(2):
                        augmented_audio = augment(samples=audio, sample_rate=sr)
                        temp_path = "temp_augmented.wav"
                        sf.write(temp_path, augmented_audio, sr)
                        if os.path.exists(temp_path):
                            feat = extract_features(temp_path)
                            if feat is not None:
                                features.append(feat)
                                labels.append(label)
    features = [f for f in features if f is not None]
    if not features:
        raise ValueError("هیچ ویژگی‌ای استخراج نشد.")
    features = np.array(features)
    labels = np.array(labels)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return features, labels

def build_model(input_dim, num_classes):
    model = Sequential()      
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, kernel_regularizer=l2(0.01)))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(64, kernel_regularizer=l2(0.01)))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(32))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def record_audio(filename="temp_record.wav", duration=5, sr=16000, silence_threshold=1e-4):
    app.log("در حال ضبط صدا...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    max_amplitude = np.max(np.abs(recording))
    if max_amplitude < silence_threshold:
        app.log("هیچ صدایی ضبط نشد! لطفاً دوباره تلاش کنید.")
        return False
    sf.write(filename, recording, sr)
    app.log("ضبط انجام شد.")
    return True

class AccentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.iconbitmap("icon.ico")

        self.title("تشخیص لهجه")
        self.geometry("600x450")
        self.resizable(False, False)

        self.btn_train = tk.Button(self, text="آموزش مدل", command=self.train_model_thread)
        self.btn_train.pack(pady=10)

        self.btn_record = tk.Button(self, text="ضبط و تشخیص لهجه", command=self.record_and_predict_thread, state='disabled')
        self.btn_record.pack(pady=10)

        self.text_log = scrolledtext.ScrolledText(self, height=15)
        self.text_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.btn_quit = tk.Button(self, text="خروج", command=self.quit)
        self.btn_quit.pack(pady=5)

        self.model = None
        self.le = None
        self.scaler = None

    def log(self, msg):
        self.text_log.insert(tk.END, msg + "\n")
        self.text_log.see(tk.END)
        self.update()

    def train_model_thread(self):
        t = threading.Thread(target=self.train_model)
        t.start()

    def train_model(self):
        try:
            self.btn_train.config(state='disabled')
            self.log("آماده‌سازی داده‌ها...")
            dataset_path = "dataset"
            X, y = prepare_dataset(dataset_path)
            if len(X) == 0:
                self.log("هیچ دیتایی پیدا نشد!")
                self.btn_train.config(state='normal')
                return

            self.log("تبدیل برچسب‌ها...")
            self.le = LabelEncoder()
            y_encoded = self.le.fit_transform(y)
            y_cat = to_categorical(y_encoded)
            with open("label_encoder.pkl", "wb") as f:
                pickle.dump(self.le, f)

            X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

            self.log("شروع Cross-Validation...")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            val_accuracies = []
            for i, (train_index, val_index) in enumerate(kf.split(X_train), 1):
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
                model = build_model(X.shape[1], y_cat.shape[1])
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(X_train_fold, y_train_fold, epochs=30, batch_size=4,
                          validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping], verbose=0)
                loss, acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
                val_accuracies.append(acc)
                self.log(f"Fold {i} accuracy: {acc:.4f}")
            self.log(f"میانگین دقت اعتبارسنجی: {np.mean(val_accuracies):.4f}")

            self.log("آموزش مدل نهایی...")
            self.model = build_model(X.shape[1], y_cat.shape[1])
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            self.model.fit(X_train, y_train, epochs=30, batch_size=2, validation_split=0.2, callbacks=[early_stopping])
            loss, acc = self.model.evaluate(X_test, y_test)
            self.log(f"دقت روی داده تست: {acc * 100:.2f}%")
            self.model.save("nn_accent_model.keras")

            with open("scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)

            self.btn_record.config(state='normal')
            self.log("آموزش پایان یافت. اکنون می‌توانید صدای خود را ضبط کنید.")
        except Exception as e:
            self.log(f"خطا: {str(e)}")
        finally:
            self.btn_train.config(state='normal')

    def record_and_predict_thread(self):
        t = threading.Thread(target=self.record_and_predict)
        t.start()

    def record_and_predict(self):
        if not self.model or not self.le or not self.scaler:
            try:
                with open("nn_accent_model.keras", "rb"):
                    pass
                self.model = tf.keras.models.load_model("nn_accent_model.keras")
                with open("label_encoder.pkl", "rb") as f:
                    self.le = pickle.load(f)
                with open("scaler.pkl", "rb") as f:
                    self.scaler = pickle.load(f)
            except:
                self.log("ابتدا باید مدل را آموزش دهید یا مدل ذخیره‌شده وجود ندارد.")
                return

        success = record_audio()
        if not success:
            return
        feat = extract_features("temp_record.wav")
        if feat is not None:
            feat = self.scaler.transform(feat.reshape(1, -1))
            pred = self.model.predict(feat)
            predicted_index = np.argmax(pred)
            label = self.le.inverse_transform([predicted_index])[0]
            confidence = np.max(pred) * 100
            self.log(f"لهجه تشخیص داده‌شده: {label} (اعتماد: {confidence:.2f}%)")
        else:
            self.log("خطا در استخراج ویژگی‌ها از صدای ضبط‌شده.")

if __name__ == "__main__":
    app = AccentApp()
    app.mainloop()
