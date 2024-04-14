import streamlit as st
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
import joblib
import os
mlp_model = joblib.load('mlp_model.joblib')
scaler = joblib.load('scaler.joblib')
le_disorder = joblib.load('le_disorder.joblib')  

suggest_df = pd.read_csv("suggest.csv")
music_folder_path = "music"

yoga_df = pd.read_csv("yoga.csv")
mental_health_yoga_df = pd.read_csv("mental_health_yoga.csv")

dss_flat = pd.read_csv("dss_flat.csv")

raw_file_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample', 'sample_audvis_raw.fif')

raw = mne.io.read_raw_fif(raw_file_path, preload=True)

evoked_file_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample', 'sample_audvis-ave.fif')
evoked = mne.read_evokeds(evoked_file_path, baseline=(None, 0), proj=True)

if not evoked:
    st.error("No evoked data loaded. Check the file path and content.")
else:
    evoked_l_aud = evoked[0]
    evoked_r_aud = evoked[1]
    evoked_l_vis = evoked[2]
    evoked_r_vis = evoked[3]

st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
            font-family: 'Arial', sans-serif;
        }
        .header {
            background-color: #3498db;
            color: #ffffff;
            padding: 10px;
            text-align: center;
            font-size: 36px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>NeuroFeedback Meditation Web App 🧠</div>", unsafe_allow_html=True)

menu = ["Home", "Prediction",  "About", "Contact"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.subheader("Welcome to the EEG Disorder Prediction App")
    st.markdown("This web app uses machine learning to predict specific disorders based on EEG data. "
                "Enter your EEG data in the form, and we will provide predictions along with helpful suggestions.")
    st.image("your_logo.png", caption="EEG BASED COMPUTATION PROGRAM", use_column_width=True)

elif choice == "Prediction":
    st.header("Enter EEG Data:")
    
    with st.form(key='user_input_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            channel = st.text_input("Channel:", "")
        with col2:
            delta = st.text_input("Delta:", "")
            theta = st.text_input("Theta:", "")
        with col3:
            alpha = st.text_input("Alpha:", "")
            beta = st.text_input("Beta:", "")
            highbeta = st.text_input("High Beta:", "")
            gamma = st.text_input("Gamma:", "")
        
        predict_button = st.form_submit_button("Predict")

    eeg_data_placeholder = st.empty()

    if predict_button:
        custom_input = pd.DataFrame([[channel, delta, theta, alpha, beta, highbeta, gamma]],
                                      columns=['channel', 'delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma'])

        encoded_channel = pd.concat([dss_flat['channel'].astype(str), custom_input['channel']]).astype(int).iloc[-1]
        custom_input['channel'] = encoded_channel
        custom_input['specific.disorder'] = 0  # Use the same value used during training

        custom_input_scaled = scaler.transform(custom_input.drop(['specific.disorder'], axis=1))

        custom_prediction = mlp_model.predict(custom_input_scaled)

        # Get the predicted class name
        predicted_class_name = le_disorder.inverse_transform(custom_prediction)[0]

        # Display the result
        if predicted_class_name == "Healthy Control":
            st.success("You are Healthy Control.")
        else:
            st.success(f"Predicted Class: {predicted_class_name}")
            st.subheader(f"We suggest you follow this video and audio procedure to manage your {predicted_class_name}.")

            video_match = suggest_df[suggest_df.apply(lambda row: predicted_class_name.lower() in row['class'].lower(), axis=1)]

            if not video_match.empty:
                video_url = video_match['url'].values[0]
                st.video(video_url)

                yoga_suggestion = yoga_df[yoga_df.apply(lambda row: predicted_class_name.lower() in row['class'].lower(), axis=1)]['url'].values[0]
                meditation_suggestion = mental_health_yoga_df[mental_health_yoga_df.apply(lambda row: predicted_class_name.lower() in row['class'].lower(), axis=1)]['Yoga'].values[0]

                st.subheader("Yoga Suggestion:")
                st.markdown(f"[{predicted_class_name} Yoga Video]({yoga_suggestion})")
                st.video(yoga_suggestion)  

                st.subheader("Meditation & Music Therpy Suggestion:")
                st.markdown(f"{meditation_suggestion}")

                st.markdown(meditation_suggestion)

            else:
                st.warning("No matching video URL found for the predicted class. Showing a default message or action.")
            audio_filename = f"{predicted_class_name.lower()}.mp3"
            audio_path = os.path.join(music_folder_path, audio_filename)

            if not os.path.exists(audio_path):
                audio_filename = f"{predicted_class_name.capitalize()}.mp3"
                audio_path = os.path.join(music_folder_path, audio_filename)

            if os.path.exists(audio_path):
                st.audio(audio_path, format="audio/mp3", start_time=0)
            else:
                st.warning(f"No matching audio file found for the predicted class ({audio_filename}). Showing a default message or action.")
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots()
                frequencies = ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']
                spectrum_values = [float(custom_input[frequency]) for frequency in frequencies]
                ax.bar(frequencies, spectrum_values, color='blue')
                ax.set_title('EEG Spectrum')
                ax.set_xlabel('Frequency Bands')
                ax.set_ylabel('Power/Frequency')
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots()
                ax.plot(frequencies, spectrum_values, marker='o', linestyle='-', color='green')
                ax.set_title('Custom EEG Plot')
                ax.set_xlabel('Frequency Bands')
                ax.set_ylabel('Power/Frequency')
                st.pyplot(fig)
                from mne_plot import generate_mne_plot
            
            def eeg_data_page_content():
                st.header("EEG Data Visualization")

                st.subheader("Evoked Data")
                evoked_plot = evoked_l_aud.plot(exclude=())
                st.pyplot(evoked_plot)

                picks = mne.pick_types(evoked_l_aud.info, meg=True, eeg=False, eog=False)

                st.subheader("Evoked Data with Spatial Colors and GFP")
                evoked_spatial_plot = evoked_l_aud.plot(spatial_colors=True, gfp=True, picks=picks)
                st.pyplot(evoked_spatial_plot)

                st.subheader("Topomaps")
                st.write("Left Auditory Topomap:")
                left_aud_topomap = evoked_l_aud.plot_topomap(times=0.1, show=False)
                st.pyplot(left_aud_topomap)

                st.write("Right Auditory Topomap:")
                right_aud_topomap = evoked_r_aud.plot_topomap(times=0.1, ch_type='mag', show=False)
                st.pyplot(right_aud_topomap)

                st.subheader("Topomaps in Subplots")
                fig, ax = plt.subplots(1, 4)

                ax_topo_1 = ax[0].inset_axes([0, 0, 0.45, 1])
                ax_topo_2 = ax[0].inset_axes([0.55, 0, 0.45, 1])
                evoked_l_aud.plot_topomap(times=0.1, axes=[ax_topo_1, ax_topo_2], show=False)
                st.pyplot(fig)

                st.subheader("Make Field Map")
                subjects_dir = op.join(mne.datasets.sample.data_path(), 'subjects')
                trans_fname = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
                maps = mne.make_field_map(evoked_l_aud, trans=trans_fname, subject='sample', subjects_dir=subjects_dir, n_jobs=1)

                st.write("Field Map at 0.1s")
                field_map_plot = evoked_l_aud.plot_field(maps, time=0.1)
                st.pyplot(field_map_plot)

        eeg_data_page_content()

        
elif choice == "About":
    st.subheader("About This Project")
    st.markdown("This web application is designed for predicting specific disorders based on EEG data using machine learning. "
                "It provides users with insights into their EEG spectrum and suggests procedures to manage predicted disorders.")
    
    st.markdown("### Key Features:")
    st.markdown("- EEG data prediction using a pre-trained machine learning model.")
    st.markdown("- Visualization of EEG spectrum with customizable plots.")
    st.markdown("- Personalized suggestions for managing predicted disorders.")
    
    st.markdown("### Technologies Used:")
    st.markdown("- Python (Streamlit for web app development)")
    st.markdown("- Machine Learning (MLP model for disorder prediction)")
    st.markdown("- Data Preprocessing (Pandas, joblib for model and scaler loading)")

elif choice == "Contact":
    st.subheader("Contact Us")
    st.markdown("If you have any questions, suggestions, or feedback, feel free to contact us.")


st.sidebar.markdown("Page navigation")

st.markdown("""
    <style>
        footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

footer = """
    <div class="footer">
        <p>2024 @COPYRIGHTS Team-8 CSM</p>
    </div>
    """

st.markdown(footer, unsafe_allow_html=True)

