import mne
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def generate_mne_plot(raw_eeg):
    fig, ax = plt.subplots(figsize=(10, 4))
    raw_eeg.plot_psd(ax=ax, fmax=50, spatial_colors=False, show=False)
    ax.set_title('EEG Power Spectral Density')
    ax.grid(True)

    st.pyplot(fig)

raw = mne.io.read_raw_fif(str(mne.datasets.sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'), preload=True)

picks = mne.pick_types(raw.info, eeg=True)

picks = np.array(picks, dtype=int)

info = mne.create_info(ch_names=[raw.ch_names[i] for i in picks], sfreq=raw.info['sfreq'], ch_types='eeg')

raw_eeg = mne.io.RawArray(raw.get_data(picks), info)

generate_mne_plot(raw_eeg)
