import sqlite3
import sounddevice as sd
import wave
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import librosa
import os
import numpy as np

class AudioRecorderApp:
    def __init__(self):
        self.recording_duration = 30  # seconds
        self.temp_file_name = "temp_audio.wav"
        self.saved_file_name = None
        self.audio_db = AudioDatabase()

    def record_audio(self, filename):
        st.write("Recording...")
        audio_data = sd.rec(int(44100 * self.recording_duration), samplerate=44100, channels=2, dtype='int16')
        sd.wait()

        save_path = f"{filename}.wav"  # Save with user-provided filename
        st.write("Saving audio to", save_path)
        with wave.open(save_path, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(audio_data.tobytes())

        st.write("Recording complete!")
        self.saved_file_name = save_path
        self.audio_db.insert_record(filename, os.path.abspath(save_path))

    def analyze_audio(self, file_path):
        # Load the audio file using librosa
        audio, sample_rate = librosa.load(file_path)

        # Extract features from the audio file
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=19)
        features = []
        for mfcc in mfccs:
            features.append(mfcc.mean())
        features = [features]

        # Standardize the features
        features = scaler.transform(features)
        # Predict the label of the audio file using the trained model
        prediction = model.predict(features)[0]

        if prediction == 0:
            st.write("The person in the audio file does not have Parkinson's disease.")
        else:
            probability = model.predict_proba(features)
            proba_accuracy = np.max(probability)
            st.write(proba_accuracy)
            if proba_accuracy < 0.3:
                st.write("The person in the audio file is affected by a low level of Parkinson's disease.")
            elif proba_accuracy < 0.7:
                st.write("The person in the audio file is affected by a medium level of Parkinson's disease.")
            else:
                st.write("The person in the audio file is affected by a high level of Parkinson's disease.")


class ParkinsonsSymptomsApp:
    def __init__(self):
        self.symptoms = {
            "Tremor": {"stage": "Early Stage",
                       "meaning": "Tremor is an involuntary shaking movement that usually begins in a limb.",
                       "diet": "Foods rich in magnesium such as spinach, nuts, seeds, and whole grains may help reduce tremors."},
            "Bradykinesia": {"stage": "Early Stage",
                             "meaning": "Bradykinesia refers to slowness of movement and is one of the cardinal symptoms of Parkinson's disease.",
                             "diet": "Consuming foods high in omega-3 fatty acids like fatty fish, flaxseeds, and walnuts may support brain health and improve movement."},
            "Rigidity": {"stage": "Early Stage", "meaning": "Rigidity is stiffness or inflexibility of the muscles.",
                         "diet": "Incorporating foods with anti-inflammatory properties like berries, leafy greens, and turmeric may help reduce rigidity."},
            "Postural instability": {"stage": "Mid Stage",
                                     "meaning": "Postural instability is difficulty maintaining an upright position and balance.",
                                     "diet": "Maintaining adequate hydration and consuming foods high in calcium and vitamin D, such as dairy products and leafy greens, may support bone health and balance."},
            "Low Blood Pressure": {"stage": "Mid Stage",
                                   "meaning": "Low blood pressure, or hypotension, is a drop in blood pressure levels that can cause dizziness and fainting.",
                                   "diet": "Drink 8 glasses of water each day, add salt to your food, and avoid alcohol."},
            "Impaired balance": {"stage": "Late Stage",
                                 "meaning": "Impaired balance is the inability to maintain a steady and stable posture.",
                                 "diet": "Eating a balanced diet with adequate protein, carbohydrates, and healthy fats, along with regular physical activity, may help improve balance and coordination."},
            "Micrographia": {"stage": "Late Stage",
                             "meaning": "Micrographia is a characteristic handwriting symptom where the writing becomes progressively smaller and more cramped.",
                             "diet": "Including foods high in antioxidants and vitamins, such as berries, citrus fruits, and dark leafy greens, may support overall brain health and motor function."},
            "Constipation": {"stage": "Early Stage",
                             "meaning": "Constipation is a common gastrointestinal symptom in Parkinson's disease.",
                             "diet": "Add more fiber to your diet."},
            "Dystonia": {"stage": "Mid Stage",
                         "meaning": "Dystonia is a movement disorder characterized by involuntary muscle contractions.",
                         "diet": "Eating small, frequent meals and avoiding trigger foods may help manage dystonia symptoms."},
            "Drooling": {"stage": "Late Stage",
                         "meaning": "Drooling, or sialorrhea, is excessive saliva production and difficulty swallowing.",
                         "diet": "Avoiding spicy foods and consuming small, frequent meals may help reduce drooling."},
            "Speech changes": {"stage": "Early Stage",
                               "meaning": "Speech changes in Parkinson's disease include soft, monotone, and slurred speech.",
                               "diet": "Maintaining hydration and consuming foods that are easy to swallow may help manage speech changes. Eating lean protein, legumes, and whole grains."},
            "Swallowing difficulties": {"stage": "Late Stage",
                                        "meaning": "Swallowing difficulties, or dysphagia, occur when it becomes hard to move food or liquid from the mouth to the stomach.",
                                        "diet": "Following a soft food diet and practicing swallowing exercises may help improve swallowing difficulties. Prefer liquid food or eat food that is soft. Avoid dry breads and cakes, rice, nuts, and seeds."}
        }
        self.selected_symptoms = set()
        self.stage_counts = {"Early Stage": 1, "Mid Stage": 2, "Late Stage": 3}

    def get_overall_stage(self):
        stages = {"Early Stage": 0, "Mid Stage": 0, "Late Stage": 0}
        for symptom in self.selected_symptoms:
            stage = self.symptoms[symptom]["stage"]
            stages[stage] += 1

        if stages["Late Stage"] > 0:
            return "Late Stage"
        elif stages["Mid Stage"] > 0:
            return "Mid Stage"
        else:
            return "Early Stage"

    def get_diet_plans(self):
        diet_plans = set()
        for symptom in self.selected_symptoms:
            diet_plans.add(self.symptoms[symptom]["diet"])
        return diet_plans


class AudioDatabase:
    def __init__(self, db_name="audio_records.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS AudioRecords
                             (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             file_name TEXT NOT NULL,
                             file_path TEXT NOT NULL)''')
        self.conn.commit()

    def insert_record(self, file_name, file_path):
        self.cursor.execute('''INSERT INTO AudioRecords (file_name, file_path)
                             VALUES (?, ?)''', (file_name, file_path))
        self.conn.commit()

    def get_audio_files(self):
        self.cursor.execute('''SELECT * FROM AudioRecords''')
        return self.cursor.fetchall()

    def close_connection(self):
        self.conn.close()


# Load the Parkinson's disease dataset from the CSV file
parkinsons_data = pd.read_csv("C:/Users/aleky/Downloads/chand study/project/parkinsons/parkinsons.csv")

# Extract the relevant features from the dataset
features = parkinsons_data.loc[:, 'MDVP:Fo(Hz)': 'PPE'].values
features = features[:, [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22]]

labels = parkinsons_data.loc[:, 'status'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an SVM model on the training set
# Train an SVM model on the training set with probability=True
model = SVC(kernel='linear', C=1, gamma='auto', probability=True)
model.fit(X_train, y_train)


if __name__ == "__main__":
    app = AudioRecorderApp()
    symptoms_app = ParkinsonsSymptomsApp()

    st.title("Parkinson's Disease Detection")
    # Audio Recording Section
    filename = st.text_input("Enter a filename for the audio")
    start_button = st.button("Start Recording")
    file_path = st.file_uploader("Select Audio File")
    analyze_button = st.button("Analyze", key="analyze_audio")

    if start_button and filename:
        app.record_audio(filename)

    if file_path and analyze_button:
        app.analyze_audio(file_path)

    # Save button
    if app.saved_file_name:
        save_button = st.button("Save")
        if save_button:
            st.write(f"Saving audio as '{filename}.wav'")
            # Your code for saving the audio file can go here

    # Display audio files from the database
    st.title("Audio Files in Database")
    audio_files = app.audio_db.get_audio_files()
    if audio_files:
        for audio_file in audio_files:
            st.write(f"File Name: {audio_file[1]}")
            st.write(f"File Path: {audio_file[2]}")
    else:
        st.write("No audio files found in the database.")

    # Parkinson's Symptoms Checker Section
    st.subheader("Parkinson's Disease Symptoms Detailed Description:")
    st.write(
        "Parkinson's disease is a neurodegenerative disorder that affects movement. Here are some common symptoms and their meanings:")
    for symptom, details in symptoms_app.symptoms.items():
        st.write(f"- {symptom}: {details['meaning']}")

    st.title("Parkinson's Disease Symptoms Checker")

    st.subheader("Symptoms of Parkinson's Disease")
    for symptom in symptoms_app.symptoms:
        if st.checkbox(symptom):
            symptoms_app.selected_symptoms.add(symptom)

    if symptoms_app.selected_symptoms:
        overall_stage = symptoms_app.get_overall_stage()
        st.write(f"Overall Stage: {overall_stage}")

        diet_plans = symptoms_app.get_diet_plans()
        st.subheader("Diet Plans for Selected Symptoms:")
        for diet_plan in diet_plans:
            st.write(diet_plan)
    else:
        st.write("No symptoms selected.")
