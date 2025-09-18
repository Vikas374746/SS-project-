"""
Signal Visualization and Classification App
Educational tool for understanding continuous and discrete-time signals
Run with: streamlit run "Signals project.py"
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import threading
import time
from collections import deque
try:
    import pyaudio  # Optional, used for microphone features
except Exception:  # ImportError or platform errors
    pyaudio = None
try:
    import speech_recognition as sr  # Optional, used for speech-to-text
except Exception:
    sr = None
import wave
import io
try:
    import librosa  # Optional, for spectrograms in some paths
except Exception:
    librosa = None
import base64
from scipy.io import savemat
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    layout="wide",
    page_title="Signal Visualization & Classification",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Modern CSS with enhanced UX design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        margin: 1rem 0 0 0;
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-right: 1px solid #e2e8f0;
    }
    
    /* Mode Selection Cards */
    .mode-card {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .mode-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    
    .mode-card.active {
        border-color: #667eea;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #f8fafc;
        padding: 0.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background: transparent;
        border-radius: 12px;
        color: #64748b;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #e2e8f0;
        color: #334155;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced Metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    .status-success {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    .status-error {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    .status-info {
        background: #dbeafe;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Control Buttons */
    .control-btn-start {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    }
    
    .control-btn-stop {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    }
    
    .control-btn-refresh {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    }
    
    /* Enhanced Selectboxes and Sliders */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chart Containers */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #f1f5f9;
        margin: 1rem 0;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        border-color: #667eea;
    }
    
    /* Loading Animation */
    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }
    
    /* Dark Mode Support */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background: #1e293b;
            border-color: #334155;
            color: #f1f5f9;
        }
        
        .feature-card {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-color: #475569;
            color: #f1f5f9;
        }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIGNAL GENERATION AND PROCESSING
# =============================================================================

class SignalProcessor:
    """Advanced signal processing operations"""
    
    @staticmethod
    def apply_filter(x, fs, filter_type, cutoff, order=4):
        """Apply digital filters to signal"""
        nyquist = fs / 2
        
        if filter_type == "lowpass":
            sos = signal.butter(order, cutoff/nyquist, btype='low', output='sos')
        elif filter_type == "highpass":
            sos = signal.butter(order, cutoff/nyquist, btype='high', output='sos')
        elif filter_type == "bandpass":
            low, high = cutoff
            sos = signal.butter(order, [low/nyquist, high/nyquist], btype='band', output='sos')
        elif filter_type == "bandstop":
            low, high = cutoff
            sos = signal.butter(order, [low/nyquist, high/nyquist], btype='bandstop', output='sos')
        else:
            return x  # No filtering
            
        filtered_signal = signal.sosfilt(sos, x)
        return filtered_signal
    
    @staticmethod
    def mix_signals(*signals, weights=None):
        """Mix multiple signals with optional weights"""
        if weights is None:
            weights = [1.0] * len(signals)
        
        # Find minimum length
        min_length = min(len(sig) for sig in signals)
        
        # Mix signals
        mixed = np.zeros(min_length)
        for sig, weight in zip(signals, weights):
            mixed += weight * sig[:min_length]
            
        return mixed
    
    @staticmethod
    def calculate_power_energy(x, fs=1000):
        """Calculate signal power and energy metrics"""
        # Signal power (average power)
        power = np.mean(x**2)
        
        # Signal energy (total energy)
        energy = np.sum(x**2) / fs
        
        # RMS value
        rms = np.sqrt(power)
        
        # Peak power
        peak_power = np.max(x**2)
        
        # Crest factor
        crest_factor = np.max(np.abs(x)) / rms if rms > 0 else 0
        
        return {
            'power': power,
            'energy': energy,
            'rms': rms,
            'peak_power': peak_power,
            'crest_factor': crest_factor
        }
    
    @staticmethod
    def generate_spectrogram(x, fs, nperseg=256):
        """Generate spectrogram data"""
        f, t, Sxx = signal.spectrogram(x, fs, nperseg=nperseg)
        return f, t, Sxx
        
    @staticmethod
    def calculate_harmonic_distortion(x, fs):
        """Calculate Total Harmonic Distortion (THD)"""
        X = fft(x)
        freqs = fftfreq(len(x), 1/fs)
        psd = np.abs(X)**2
        
        # Find fundamental frequency (highest peak)
        half_idx = len(psd) // 2
        fundamental_idx = np.argmax(psd[:half_idx])
        fundamental_freq = freqs[fundamental_idx]
        fundamental_power = psd[fundamental_idx]
        
        # Find harmonics (2x, 3x, 4x, 5x fundamental frequency)
        harmonic_powers = []
        for i in range(2, 6):
            harmonic_freq = fundamental_freq * i
            # Find closest frequency bin
            harmonic_idx = np.argmin(np.abs(freqs[:half_idx] - harmonic_freq))
            harmonic_powers.append(psd[harmonic_idx])
        
        # Calculate THD
        thd = np.sqrt(np.sum(harmonic_powers)) / np.sqrt(fundamental_power) if fundamental_power > 0 else 0
        
        return {
            'fundamental_freq': fundamental_freq,
            'fundamental_power': fundamental_power,
            'harmonic_powers': harmonic_powers,
            'thd': thd
        }
    
    @staticmethod
    def calculate_snr(signal, noise=None):
        """Calculate Signal-to-Noise Ratio (SNR)"""
        if noise is not None:
            # If noise is provided separately
            signal_power = np.mean(signal**2)
            noise_power = np.mean(noise**2)
        else:
            # Estimate noise from signal (assuming signal is mostly noise-free)
            # Use median absolute deviation as robust noise estimator
            signal_power = np.mean(signal**2)
            noise_estimate = np.median(np.abs(signal - np.median(signal))) * 1.4826
            noise_power = noise_estimate**2
        
        # Calculate SNR in dB
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        return snr

class SignalExporter:
    """Export signals and plots in various formats"""
    
    @staticmethod
    def export_to_csv(t, x, filename="signal_data.csv"):
        """Export signal data to CSV with validation"""
        min_len = min(len(t), len(x))
        t_export = np.asarray(t[:min_len]).reshape(-1)
        x_export = np.asarray(x[:min_len]).reshape(-1)
        df = pd.DataFrame({'Time': t_export, 'Amplitude': x_export, 'Sample_Index': np.arange(min_len)})
        return df.to_csv(index=False)
    
    @staticmethod
    def export_to_mat(t, x, filename="signal_data.mat"):
        """Export signal data to MATLAB format with validation"""
        min_len = min(len(t), len(x))
        t_export = np.asarray(t[:min_len]).reshape(-1)
        x_export = np.asarray(x[:min_len]).reshape(-1)
        data = {
            'time': t_export,
            'amplitude': x_export,
            'num_samples': min_len
        }
        if min_len > 1 and float(t_export[-1]) != float(t_export[0]):
            data['sampling_rate'] = len(t_export) / (float(t_export[-1]) - float(t_export[0]))
            data['duration'] = float(t_export[-1]) - float(t_export[0])
        else:
            data['sampling_rate'] = 1000.0
            data['duration'] = float(min_len)
        buffer = io.BytesIO()
        savemat(buffer, data)
        buffer.seek(0)
        return buffer.getvalue()
    
    @staticmethod
    def export_to_png(t, x, domain="Continuous-Time", sig_type="signal", processed_x=None):
        """Export signal plot to PNG"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot original signal
        if domain == "Continuous-Time":
            ax.plot(t, x, 'b-', linewidth=2, label='Original Signal')
            ax.set_xlabel('Time (s)')
        else:
            ax.plot(t, x, 'bo-', linewidth=2, markersize=6, label='Original Signal')
            ax.set_xlabel('Sample Index (n)')
        
        # Plot processed signal if available
        if processed_x is not None:
            if domain == "Continuous-Time":
                ax.plot(t, processed_x, 'r--', linewidth=2, label='Processed Signal')
            else:
                ax.plot(t, processed_x, 'ro--', linewidth=2, markersize=4, label='Processed Signal')
        
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{domain} Signal: {sig_type.title()}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)  # Important: close figure to free memory
        return buffer.getvalue()
    
    @staticmethod
    def create_download_link(data, filename, file_type="csv"):
        """Create download link for data"""
        if file_type == "csv":
            b64 = base64.b64encode(data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
        elif file_type == "mat":
            b64 = base64.b64encode(data).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'
        elif file_type == "png":
            b64 = base64.b64encode(data).decode()
            href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download {filename}</a>'
        return href

class SignalGenerator:
    """Generate various continuous and discrete-time signals"""
    
    @staticmethod
    def generate_continuous_signal(signal_type: str, params: dict) -> tuple:
        """Generate continuous-time signals"""
        fs = params.get('fs', 1000)
        duration = params.get('duration', 2.0)
        amplitude = params.get('amplitude', 1.0)
        frequency = params.get('frequency', 10)
        phase = params.get('phase', 0)
        noise_level = params.get('noise_level', 0.0)

        # Ensure sufficient temporal resolution for smooth curves:
        # at least 64 samples per cycle when frequency is high.
        base_samples = int(fs * duration)
        min_per_cycle = 64
        dynamic_samples = int(max(base_samples, min_per_cycle * max(1.0, frequency) * duration))
        t = np.linspace(0, duration, dynamic_samples, endpoint=False)

        if signal_type == "sine":
            x = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        elif signal_type == "cosine":
            x = amplitude * np.cos(2 * np.pi * frequency * t + phase)
        elif signal_type == "square":
            duty_cycle = params.get('duty_cycle', 0.5)
            x = amplitude * signal.square(2 * np.pi * frequency * t + phase, duty=duty_cycle)
        elif signal_type == "sawtooth":
            # High-quality sawtooth using scipy; width=1 for rising ramp
            x = amplitude * signal.sawtooth(2 * np.pi * frequency * t + phase, width=1)
        elif signal_type == "triangle":
            # Triangle wave via symmetric sawtooth
            x = amplitude * signal.sawtooth(2 * np.pi * frequency * t + phase, width=0.5)
        elif signal_type == "ramp":
            # Generate linear ramp over duration
            x = amplitude * (t / duration)
            # Reset ramp for each period if frequency > 0
            if frequency > 0:
                period = 1.0 / frequency
                x = amplitude * ((t % period) / period)
        elif signal_type == "chirp":
            f1 = frequency
            f2 = frequency * 3
            x = amplitude * signal.chirp(t, f1, duration, f2)
        elif signal_type == "exponential":
            decay = params.get('decay', 1.0)
            x = amplitude * np.exp(-decay * t) * np.sin(2 * np.pi * frequency * t)
        elif signal_type == "gaussian_pulse":
            sigma = params.get('sigma', 0.1)
            x = amplitude * np.exp(-(t - duration/2)**2 / (2 * sigma**2))
        else:  # default to sine
            x = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        
        # Add noise if specified
        if noise_level > 0:
            x += np.random.normal(0, noise_level, len(t))
            
        return t, x
    
    @staticmethod
    def generate_discrete_signal(signal_type: str, params: dict) -> tuple:
        """Generate discrete-time signals"""
        n_samples = params.get('n_samples', 50)
        amplitude = params.get('amplitude', 1.0)
        frequency = params.get('frequency', 0.1)  # Digital frequency (cycles per sample)
        phase = params.get('phase', 0)
        noise_level = params.get('noise_level', 0.0)
        duty_cycle = params.get('duty_cycle', 0.5)
        
        # Increase temporal resolution for discrete visualization by oversampling, then sampling on integers
        n = np.arange(n_samples)
        
        if signal_type == "sine":
            x = amplitude * np.sin(2 * np.pi * frequency * n + phase)
        elif signal_type == "cosine":
            x = amplitude * np.cos(2 * np.pi * frequency * n + phase)
        elif signal_type == "square":
            # Proper discrete square wave with duty cycle
            x = amplitude * signal.square(2 * np.pi * frequency * n + phase, duty=duty_cycle)
        elif signal_type == "sawtooth":
            # Discrete sawtooth (ramp) using scipy with width=1 for proper ramp
            x = amplitude * signal.sawtooth(2 * np.pi * frequency * n + phase, width=1)
        elif signal_type == "triangle":
            # Discrete triangle using sawtooth with width=0.5
            x = amplitude * signal.sawtooth(2 * np.pi * frequency * n + phase, width=0.5)
        elif signal_type == "exponential":
            decay = params.get('decay', 0.1)
            x = amplitude * np.power(1 - decay, n)
        elif signal_type == "step":
            step_point = params.get('step_point', n_samples // 2)
            x = np.zeros(n_samples)
            x[step_point:] = amplitude
        elif signal_type == "impulse":
            impulse_point = params.get('impulse_point', n_samples // 2)
            x = np.zeros(n_samples)
            x[impulse_point] = amplitude
        elif signal_type == "ramp":
            # Periodic ramp if frequency > 0, else single ramp
            if frequency > 0:
                period = max(1, int(round(1.0 / frequency)))
                x = amplitude * ((n % period) / (period - 1 if period > 1 else 1))
            else:
                x = amplitude * n / max(1, n_samples - 1)
        elif signal_type == "random":
            x = amplitude * np.random.randn(n_samples)
        else:  # default to sine
            x = amplitude * np.sin(2 * np.pi * frequency * n + phase)
        
        # Add noise if specified
        if noise_level > 0:
            x += np.random.normal(0, noise_level, len(n))
            
        return n, x

class RealTimeSignalGenerator:
    """Generate real-time streaming signals"""
    
    def __init__(self):
        self.is_running = False
        self.buffer_size = 1000
        self.sample_rate = 100  # Hz
        self.buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.start_time = None
        self._last_sim_time = 0.0
    def start_streaming(self, signal_params):
        """Start real-time signal generation"""
        self.is_running = True
        self.signal_params = signal_params
        self.start_time = time.time()
        self._last_sim_time = 0.0
        
    def stop_streaming(self):
        """Stop real-time signal generation"""
        self.is_running = False
        
    def _sample_at(self, t_seconds: float) -> float:
        signal_type = self.signal_params.get('type', 'sine')
        amplitude = self.signal_params.get('amplitude', 2.0)
        frequency = self.signal_params.get('frequency', 2.0)
        noise_level = self.signal_params.get('noise_level', 0.0)
        phase = self.signal_params.get('phase', 0.0)
        if amplitude == 0:
            amplitude = 2.0
        if frequency == 0:
            frequency = 2.0
        if signal_type == 'sine':
            val = amplitude * np.sin(2 * np.pi * frequency * t_seconds + phase)
        elif signal_type == 'cosine':
            val = amplitude * np.cos(2 * np.pi * frequency * t_seconds + phase)
        elif signal_type == 'square':
            t_mod = (frequency * t_seconds) % 1.0
            val = amplitude * (2 * (t_mod >= 0.5) - 1)
        elif signal_type == 'sawtooth':
            t_mod = (frequency * t_seconds) % 1.0
            val = amplitude * (2 * t_mod - 1)
        elif signal_type == 'triangle':
            t_mod = (frequency * t_seconds) % 1.0
            val = amplitude * (4 * t_mod - 1 if t_mod < 0.5 else 3 - 4 * t_mod)
        elif signal_type == 'noise':
            val = amplitude * np.random.randn()
        elif signal_type == 'chirp':
            f_end = frequency * 3
            inst_f = frequency + (f_end - frequency) * t_seconds / 10
            val = amplitude * np.sin(2 * np.pi * inst_f * t_seconds + phase)
        else:
            val = amplitude * np.sin(2 * np.pi * frequency * t_seconds + phase)
        if noise_level > 0:
            val += np.random.normal(0, noise_level)
        return float(val)

    def advance(self):
        """Advance the generator to current wall time, generating multiple samples if needed."""
        if not self.is_running or self.start_time is None:
            return None, None
        now_t = time.time() - self.start_time
        dt = 1.0 / float(self.sample_rate)
        steps = 0
        max_steps = int(self.sample_rate * 2)
        last_val = None
        while self._last_sim_time + 1e-9 <= now_t and steps < max_steps:
            t = self._last_sim_time
            val = self._sample_at(t)
            self.buffer.append(val)
            self.time_buffer.append(t)
            self._last_sim_time += dt
            steps += 1
            last_val = val
        if last_val is None:
            # No new sample; still return latest
            if self.time_buffer:
                return self.time_buffer[-1], self.buffer[-1]
            return None, None
        return self.time_buffer[-1], last_val
    
    def get_buffer_data(self):
        """Get current buffer data"""
        return list(self.time_buffer), list(self.buffer)

class MicrophoneSignalCapture:
    """Capture and process microphone input for signal analysis"""
    
    def __init__(self):
        self.is_recording = False
        self.sample_rate = 16000  # Standard for speech
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16 if pyaudio else None
        self.audio_buffer = deque(maxlen=16000)  # 1 second buffer
        self.time_buffer = deque(maxlen=16000)
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.audio = None
        self.stream = None
        
        # Configure recognizer for better performance
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
        
    def start_recording(self):
        """Start microphone recording"""
        try:
            if pyaudio is None:
                st.error("PyAudio not installed. Microphone features are disabled.")
                return False
            if self.audio is None:
                self.audio = pyaudio.PyAudio()
            
            # Check if microphone is available
            device_count = self.audio.get_device_count()
            if device_count == 0:
                st.error("No audio devices found")
                return False
            
            # Find default input device
            default_input = None
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    default_input = i
                    break
            
            if default_input is None:
                st.error("No input audio device found")
                return False
            
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=default_input,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_recording = True
            # Clear buffers
            self.audio_buffer.clear()
            self.time_buffer.clear()
            return True
            
        except Exception as e:
            st.error(f"Microphone initialization error: {str(e)}")
            self.is_recording = False
            return False
    
    def stop_recording(self):
        """Stop microphone recording"""
        self.is_recording = False
        try:
            if hasattr(self, 'stream') and self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            if hasattr(self, 'audio') and self.audio is not None:
                self.audio.terminate()
                self.audio = None
        except Exception as e:
            st.warning(f"Error stopping microphone: {str(e)}")
    
    def get_audio_chunk(self):
        """Get next audio chunk from microphone"""
        if not self.is_recording or not hasattr(self, 'stream') or self.stream is None:
            return None, None
            
        try:
            # Handle potential overflow by catching the exception
            try:
                data = self.stream.read(self.chunk_size)
            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    # Input overflow - just skip this chunk and continue
                    return list(self.time_buffer), list(self.audio_buffer)
                else:
                    raise e
            
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Normalize to [-1, 1]
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Add to buffer
            current_time = time.time()
            for i, sample in enumerate(audio_data):
                self.audio_buffer.append(sample)
                self.time_buffer.append(current_time + i / self.sample_rate)
            
            return list(self.time_buffer), list(self.audio_buffer)
        except Exception as e:
            # Don't show error for every audio read failure to avoid spam
            return list(self.time_buffer), list(self.audio_buffer)
    
    def get_buffer_data(self):
        """Get current buffer data"""
        return list(self.time_buffer), list(self.audio_buffer)
    
    def recognize_speech(self, audio_data):
        """Perform speech recognition on audio data"""
        if sr is None:
            return "Speech recognition unavailable (missing speech_recognition)"
        try:
            if len(audio_data) < 1600:
                return "Insufficient audio data"
            audio_int16 = (np.array(audio_data) * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            audio_data_obj = sr.AudioData(audio_bytes, self.sample_rate, 2)
            text = self.recognizer.recognize_google(audio_data_obj, language='en-US')
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Speech recognition service error: {str(e)}"
        except sr.WaitTimeoutError:
            return "Speech recognition timeout"
        except Exception as e:
            return f"Audio processing error: {str(e)}"

class SignalClassifier:
    """Enhanced signal classifier with symmetry, nature, and energy analysis"""

    @staticmethod
    def check_symmetry(x):
        """Check if signal is even, odd, or neither"""
        # Create reversed signal for comparison
        x_rev = x[::-1]
        # Check even symmetry: x[n] = x[-n]
        even_error = np.mean(np.abs(x - x_rev))
        # Check odd symmetry: x[n] = -x[-n]
        odd_error = np.mean(np.abs(x + x_rev))
        # Use threshold for numerical stability
        threshold = 1e-10 * np.max(np.abs(x))
        
        if even_error < threshold:
            return "Even"
        elif odd_error < threshold:
            return "Odd"
        return "Neither"

    @staticmethod
    def check_deterministic(x, t_or_n):
        """Check if signal appears deterministic or random"""
        # Compute signal differences and their statistics
        diff = np.diff(x)
        std_ratio = np.std(diff) / (np.std(x) + 1e-12)
        
        # Check for patterns in the signal
        fft_x = np.abs(fft(x))
        peak_ratio = np.max(fft_x) / np.mean(fft_x)
        
        # Analyze local predictability
        predictability = 1.0 - (std_ratio / (1 + peak_ratio))
        
        return "Deterministic" if predictability > 0.7 else "Random"

    @staticmethod
    def analyze_energy_power(x, t_or_n, is_discrete):
        """Classify signal as energy or power signal"""
        if is_discrete:
            energy = np.sum(np.abs(x)**2)
            power = energy / len(x)  # Average power
        else:
            dt = t_or_n[1] - t_or_n[0]
            energy = np.sum(np.abs(x)**2) * dt
            power = energy / (t_or_n[-1] - t_or_n[0])
        
        # Classification logic
        if energy < 1e6:  # Finite energy threshold
            return "Energy Signal"
        elif 0.01 < power < 1e6:  # Reasonable power range
            return "Power Signal"
        return "Undefined"

    @staticmethod
    def detect_elementary_signal(x, t_or_n):
        """Identify basic signal types"""
        # Normalize signal for consistent detection
        x_norm = (x - np.mean(x)) / (np.std(x) + 1e-12)
        
        # Check for sudden jump (unit step)
        if np.any(np.abs(np.diff(x_norm)) > 3.0) and len(np.where(np.abs(np.diff(x_norm)) > 1.0)[0]) < 3:
            return "Unit Step"
        
        # Check for linear growth (ramp)
        diff1 = np.diff(x)
        if np.std(np.diff(diff1)) / (np.std(diff1) + 1e-12) < 0.1:
            return "Ramp"
        
        # Check for impulse
        if np.max(np.abs(x_norm)) > 5.0 and np.sum(np.abs(x_norm) > 2.0) < len(x_norm) * 0.1:
            return "Impulse"
        
        # Check for sinusoidal
        fft_x = np.abs(fft(x_norm))
        if np.max(fft_x[1:]) / np.mean(fft_x[1:]) > 10:
            return "Sinusoidal"
        
        # Check for exponential
        log_x = np.log(np.abs(x - np.min(x) + 1e-12))
        if np.std(np.diff(log_x)) / (np.std(log_x) + 1e-12) < 0.2:
            return "Exponential"
        
        return "Complex/Other"

    @staticmethod
    def extract_features(t_or_n, x, is_discrete=False):
        features = {}
        # Store signal for length calculations
        features['signal'] = x
        
        # Basic statistics with improved robustness
        features['mean'] = float(np.mean(x))
        features['std'] = float(np.std(x))
        features['rms'] = float(np.sqrt(np.mean(np.square(x))))
        features['max'] = float(np.max(x))
        features['min'] = float(np.min(x))
        
        # Normalized statistics for better comparison
        if np.std(x) != 0:
            features['skew'] = float(np.mean((x - np.mean(x)) ** 3) / (np.std(x) ** 3))
            features['kurtosis'] = float(np.mean((x - np.mean(x)) ** 4) / (np.std(x) ** 4))
        else:
            features['skew'] = 0.0
            features['kurtosis'] = 0.0
        
        # Zero crossings with improved accuracy
        zero_crossings = np.where(np.diff(np.signbit(x)))[0]
        features['zero_crossings'] = len(zero_crossings)
        
        # Zero crossing regularity (important for periodic signals)
        if len(zero_crossings) > 1:
            zero_crossing_intervals = np.diff(zero_crossings)
            features['zero_crossing_regularity'] = np.std(zero_crossing_intervals) / np.mean(zero_crossing_intervals)
        else:
            features['zero_crossing_regularity'] = float('inf')
        
        # Peak analysis
        features['peak_to_peak'] = float(np.max(x) - np.min(x))
        
        # Spectral analysis
        X = np.abs(fft(x))
        # Normalize FFT by signal length for consistent comparison
        X = X / len(x)
        
        # Improved spectral flatness calculation
        log_spectrum = np.log(X + 1e-12)
        features['spectral_flatness'] = float(np.exp(np.mean(log_spectrum)) / (np.mean(X) + 1e-12))
        
        # Peak frequency analysis
        freqs = fftfreq(len(x), 1.0 if is_discrete else (t_or_n[1] - t_or_n[0]))
        main_freq_idx = np.argmax(X[1:len(X)//2]) + 1  # Skip DC component
        features['main_freq'] = float(abs(freqs[main_freq_idx]))
        
        # Harmonic analysis
        harmonics = X[1:len(X)//2]
        sorted_harmonics = np.sort(harmonics)[::-1]
        features['harmonic_ratio'] = float(sorted_harmonics[0] / (np.mean(sorted_harmonics[1:10]) + 1e-12))
        
        # Advanced classification features
        features['symmetry'] = SignalClassifier.check_symmetry(x)
        features['nature'] = SignalClassifier.check_deterministic(x, t_or_n)
        features['energy_power'] = SignalClassifier.analyze_energy_power(x, t_or_n, is_discrete)
        features['elementary_type'] = SignalClassifier.detect_elementary_signal(x, t_or_n)
        
        return features

    @staticmethod
    def classify_signal_type(features):
        """Enhanced signal classification with improved accuracy and detailed characteristics"""
        signal_types = []
        
        # Normalize key features for robust classification
        signal_length = len(features['signal']) if 'signal' in features else 1
        zero_crossings_per_length = features['zero_crossings'] / signal_length
        peak_to_rms = features['peak_to_peak'] / (features['rms'] + 1e-6)
        
        # 1. Primary Signal Type Detection
        if features['zero_crossings'] == 0 and features['std'] < 0.05:
            signal_types.append("DC/Constant")
            
        elif (0.3 < zero_crossings_per_length < 0.7 and
              1.5 < peak_to_rms < 3.5 and
              abs(features['skew']) < 0.3 and
              abs(features['kurtosis'] - 1.5) < 1.0):
            if features['harmonic_ratio'] > 10:  # Strong fundamental frequency
                signal_types.append(f"Sinusoidal (f≈{features['main_freq']:.2f}Hz)")
            else:
                signal_types.append("Complex Periodic")
                
        elif (features['zero_crossings'] <= 2 and 
              features['std'] > 0.1 and 
              features['kurtosis'] > 3.0):
            signal_types.append("Step")
            
        elif (features['zero_crossings'] < 3 and 
              abs(features['skew']) > 0.5 and 
              features['spectral_flatness'] < 0.1):
            signal_types.append("Ramp")
            
        elif (peak_to_rms > 3.5 and 
              features['kurtosis'] > 2.5 and 
              features['spectral_flatness'] < 0.3):
            signal_types.append("Square Wave")
            
        elif (0.2 < zero_crossings_per_length < 0.4 and 
              1.5 < peak_to_rms < 4.0 and 
              abs(features['skew']) > 0.2):
            signal_types.append("Sawtooth/Triangle")
            
        elif features['spectral_flatness'] > 0.7:
            signal_types.append("Random/Noise")
            
        else:
            signal_types.append("Complex")
        
        # 2. Symmetry Classification
        if abs(features['skew']) < 0.1:
            signal_types.append("Even Symmetric")
        elif abs(features['kurtosis']) < 0.1:
            signal_types.append("Odd Symmetric")
        else:
            signal_types.append("Non-symmetric")
        
        # 3. Signal Nature Classification
        if features['zero_crossing_regularity'] < 0.1 and features['harmonic_ratio'] > 5:
            signal_types.append("Deterministic")
        elif features['spectral_flatness'] > 0.6:
            signal_types.append("Random")
        else:
            signal_types.append("Mixed")
        
        # 4. Energy/Power Classification
        energy_power = features['energy_power']
        if isinstance(energy_power, str):
            signal_types.append(energy_power.capitalize())
        
        return signal_types
            
        # Default case
        return "Other"

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Enhanced header with modern design
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Signal Visualization & Classification</h1>
        <p>Advanced Interactive Platform for Signal Analysis, Real-Time Processing & Speech Recognition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_signal' not in st.session_state:
        st.session_state.current_signal = None
    if 'realtime_generator' not in st.session_state:
        st.session_state.realtime_generator = RealTimeSignalGenerator()
    if 'comparison_mode' not in st.session_state:
        st.session_state.comparison_mode = False
    if 'mic_capture' not in st.session_state:
        st.session_state.mic_capture = MicrophoneSignalCapture()
    if 'mixing_signals' not in st.session_state:
        st.session_state.mixing_signals = []
    
    # Initialize all session state counters to prevent AttributeError
    if 'update_counter' not in st.session_state:
        st.session_state.update_counter = 0
    if 'comp_update_counter' not in st.session_state:
        st.session_state.comp_update_counter = 0
    if 'mic_update_counter' not in st.session_state:
        st.session_state.mic_update_counter = 0
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()
    if 'comp_last_update' not in st.session_state:
        st.session_state.comp_last_update = time.time()
    if 'mic_last_update' not in st.session_state:
        st.session_state.mic_last_update = time.time()
    
    # Enhanced sidebar with modern design
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <h2 style="color: #667eea; font-weight: 600; margin: 0;">🎛️ Control Center</h2>
            <p style="color: #64748b; font-size: 0.9rem; margin: 0.5rem 0 0 0;">Configure your signal analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced mode selection with cards
        st.markdown("### 📊 Analysis Mode")
        
        mode_options = {
            "Generated Signals": {"icon": "🔧", "desc": "Create synthetic signals"},
            "Real-Time Signals": {"icon": "⚡", "desc": "Live signal streaming"},
            "Signal Comparison": {"icon": "🔄", "desc": "Compare multiple signals"},
            "Signal Mixing": {"icon": "🎵", "desc": "Mix and blend multiple signals"},
            "Microphone Input": {"icon": "🎤", "desc": "Live audio capture & speech"}
        }
        
        analysis_mode = st.selectbox(
            "Choose Analysis Mode",
            list(mode_options.keys()),
            format_func=lambda x: f"{mode_options[x]['icon']} {x}"
        )
        
        # Show mode description
        st.markdown(f"""
        <div class="status-indicator status-info" style="margin: 0.5rem 0;">
            {mode_options[analysis_mode]['icon']} {mode_options[analysis_mode]['desc']}
        </div>
        """, unsafe_allow_html=True)
        
        if analysis_mode == "Real-Time Signals":
            st.markdown("### ⚡ Real-Time Signal Parameters")
            rt_signal_type = st.selectbox("Real-Time Signal Type", ["sine", "cosine", "square", "sawtooth", "triangle", "noise", "chirp"])
            rt_amplitude = st.slider("RT Amplitude", 0.5, 10.0, 3.0, 0.1, help="Signal amplitude (higher = more visible)")
            rt_frequency = st.slider("RT Frequency (Hz)", 0.1, 5.0, 1.5, 0.1, help="Signal frequency")
            rt_phase = st.slider("RT Phase (radians)", 0.0, 2*np.pi, 0.0, 0.1, help="Phase shift")
            rt_noise_level = st.slider("RT Noise Level", 0.0, 1.0, 0.1, 0.05, help="Add random noise")
            
            rt_params = {
                'type': rt_signal_type,
                'amplitude': rt_amplitude,
                'frequency': rt_frequency,
                'phase': rt_phase,
                'noise_level': rt_noise_level
            }
            
            # Show current parameters
            with st.expander("📋 Current Parameters", expanded=False):
                st.json(rt_params)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Start Real-Time", key="rt_start", help="Begin real-time signal generation", type="primary"):
                    st.session_state.realtime_generator.start_streaming(rt_params)
                    st.success(f"🟢 Started {rt_signal_type} signal (A={rt_amplitude}, f={rt_frequency}Hz)")
                    st.rerun()
            with col2:
                if st.button("⏹️ Stop Real-Time", key="rt_stop", help="Stop real-time signal generation"):
                    st.session_state.realtime_generator.stop_streaming()
                    st.info("⏸️ Real-time streaming stopped")
                    st.rerun()
                    
        elif analysis_mode == "Microphone Input":
            st.markdown("### 🎤 Microphone Signal Capture")
            st.info("Capture live audio from your microphone for signal analysis and speech recognition.")
            
            # Enhanced microphone controls
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎙️ Start Recording", key="mic_start", help="Begin microphone audio capture"):
                    if st.session_state.mic_capture.start_recording():
                        st.success("🟢 Microphone recording started!")
                        st.rerun()
                    else:
                        if pyaudio is None:
                            st.error("❌ PyAudio not installed. Install 'pyaudio' to enable microphone.")
                        else:
                            st.error("❌ Failed to start microphone")
            with col2:
                if st.button("⏹️ Stop Recording", key="mic_stop", help="Stop microphone recording"):
                    st.session_state.mic_capture.stop_recording()
                    st.info("⏸️ Recording stopped")
                    st.rerun()
            
            # Enhanced audio settings with modern toggles
            st.markdown("#### 🎚️ Audio Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                enable_speech_recognition = st.toggle("🗣️ Speech Recognition", value=True, help="Enable real-time speech-to-text")
            with col2:
                show_audio_features = st.toggle("📊 Audio Features", value=True, help="Show detailed audio analysis")
            
        else:
            # For both "Generated Signals" and "Signal Comparison" modes
            if analysis_mode == "Signal Comparison":
                st.markdown("### 🔄 Comparison Mode")
                st.info("Configure both generated and real-time signals below, then view the comparison.")
                
                # Real-time signal controls for comparison
                st.markdown("#### Real-Time Signal")
                rt_signal_type = st.selectbox("RT Signal Type", ["sine", "square", "sawtooth", "noise", "chirp"], key="comp_rt_type")
                rt_amplitude = st.slider("RT Amplitude", 0.1, 5.0, 1.0, 0.1, key="comp_rt_amp")
                rt_frequency = st.slider("RT Frequency (Hz)", 0.1, 10.0, 1.0, 0.1, key="comp_rt_freq")
                rt_noise_level = st.slider("RT Noise Level", 0.0, 1.0, 0.0, 0.05, key="comp_rt_noise")
                
                rt_params = {
                    'type': rt_signal_type,
                    'amplitude': rt_amplitude,
                    'frequency': rt_frequency,
                    'noise_level': rt_noise_level
                }
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("▶️ Start RT for Comparison"):
                        st.session_state.realtime_generator.start_streaming(rt_params)
                        st.rerun()
                with col2:
                    if st.button("⏹️ Stop RT"):
                        st.session_state.realtime_generator.stop_streaming()
                        st.rerun()
                
                st.markdown("#### Generated Signal")
            
            # Signal type selection - ensure this is always defined
            signal_domain = st.selectbox("Signal Domain", ["Continuous-Time", "Discrete-Time"])
        
        # Only access signal_domain if it's defined (not in Real-Time or Microphone modes)
        if analysis_mode not in ["Real-Time Signals", "Microphone Input"] and signal_domain == "Continuous-Time":
            signal_types = ["sine", "cosine", "square", "sawtooth", "triangle", "chirp", "exponential", "gaussian_pulse"]
            st.markdown("### Continuous Signal Parameters")
            signal_type = st.selectbox("Signal Type", signal_types)
            
            # Common parameters
            amplitude = st.slider("Amplitude", 0.1, 5.0, 1.0, 0.1)
            frequency = st.slider("Frequency (Hz)", 0.1, 50.0, 10.0, 0.1)
            duration = st.slider("Duration (s)", 0.5, 10.0, 2.0, 0.1)
            fs = st.slider("Sampling Rate (Hz)", 100, 2000, 1000, 50)
            phase = st.slider("Phase (radians)", 0.0, 2*np.pi, 0.0, 0.1)
            noise_level = st.slider("Noise Level", 0.0, 1.0, 0.0, 0.05)
            
            # Special parameters for specific signals
            params = {
                'fs': fs, 'duration': duration, 'amplitude': amplitude,
                'frequency': frequency, 'phase': phase, 'noise_level': noise_level
            }
            
            if signal_type == "exponential":
                params['decay'] = st.slider("Decay Rate", 0.1, 5.0, 1.0, 0.1)
            elif signal_type == "gaussian_pulse":
                params['sigma'] = st.slider("Sigma", 0.01, 0.5, 0.1, 0.01)
            elif signal_type == "square":
                params['duty_cycle'] = st.slider("Duty Cycle", 0.05, 0.95, 0.5, 0.05)
            
            if st.button("Generate Continuous Signal"):
                t, x = SignalGenerator.generate_continuous_signal(signal_type, params)
                st.session_state.current_signal = (t, x, signal_domain, signal_type, params)
                
        elif analysis_mode not in ["Real-Time Signals", "Microphone Input"]:  # Discrete-Time
            signal_types = ["sine", "cosine", "square", "sawtooth", "triangle", "exponential", "step", "impulse", "ramp", "random"]
            st.markdown("### Discrete Signal Parameters")
            signal_type = st.selectbox("Signal Type", signal_types)
            
            # Common parameters
            n_samples = st.slider("Number of Samples", 10, 200, 50, 5)
            amplitude = st.slider("Amplitude", 0.1, 5.0, 1.0, 0.1)
            frequency = st.slider("Digital Frequency (cycles/sample)", 0.01, 0.5, 0.1, 0.01)
            phase = st.slider("Phase (radians)", 0.0, 2*np.pi, 0.0, 0.1)
            noise_level = st.slider("Noise Level", 0.0, 1.0, 0.0, 0.05)
            
            params = {
                'n_samples': n_samples, 'amplitude': amplitude,
                'frequency': frequency, 'phase': phase, 'noise_level': noise_level
            }
            
            # Special parameters
            if signal_type == "exponential":
                params['decay'] = st.slider("Decay Rate", 0.01, 0.5, 0.1, 0.01)
            elif signal_type == "step":
                params['step_point'] = st.slider("Step Point", 0, n_samples-1, n_samples//2, 1)
            elif signal_type == "impulse":
                params['impulse_point'] = st.slider("Impulse Point", 0, n_samples-1, n_samples//2, 1)
            elif signal_type == "square":
                params['duty_cycle'] = st.slider("Duty Cycle", 0.05, 0.95, 0.5, 0.05)
            
            if st.button("Generate Discrete Signal"):
                n, x = SignalGenerator.generate_discrete_signal(signal_type, params)
                st.session_state.current_signal = (n, x, signal_domain, signal_type, params)
    
    # Main content area
    if analysis_mode == "Real-Time Signals":
        show_realtime_interface()
    elif analysis_mode == "Signal Comparison":
        show_comparison_interface()
    elif analysis_mode == "Signal Mixing":
        show_signal_mixing_interface()
    elif analysis_mode == "Microphone Input":
        show_microphone_interface()
    elif st.session_state.current_signal:
        t_or_n, x, domain, sig_type, params = st.session_state.current_signal
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Signal Visualization", "🔍 Frequency Analysis", "📊 Signal Properties", "🤖 Classification"])
        
        with tab1:
            show_signal_visualization(t_or_n, x, domain, sig_type)
        
        with tab2:
            show_frequency_analysis(t_or_n, x, domain)
        
        with tab3:
            show_signal_properties(t_or_n, x, domain)
        
        with tab4:
            show_classification(t_or_n, x, domain, sig_type)
    
    else:
        st.info("👈 Please configure and generate a signal using the sidebar controls.")

def show_signal_mixing_interface():
    st.markdown("## 🎵 Signal Mixing & Blending")
    st.markdown("Create complex signals by mixing multiple signal types with different weights")
    
    # Initialize mixing signals in session state
    if 'mixing_signals' not in st.session_state:
        st.session_state.mixing_signals = []
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔧 Signal Configuration")
        
        # Signal generation controls
        with st.expander("➕ Add New Signal", expanded=True):
            signal_type = st.selectbox("Signal Type", ["sine", "cosine", "square", "sawtooth", "triangle", "chirp", "noise"])
            amplitude = st.slider("Amplitude", 0.1, 5.0, 1.0, 0.1)
            frequency = st.slider("Frequency (Hz)", 0.1, 50.0, 10.0, 0.1)
            phase = st.slider("Phase (radians)", 0.0, 2*np.pi, 0.0, 0.1)
            weight = st.slider("Mixing Weight", 0.0, 2.0, 1.0, 0.1)
            
            if st.button("➕ Add Signal"):
                signal_params = {
                    'type': signal_type,
                    'amplitude': amplitude,
                    'frequency': frequency,
                    'phase': phase,
                    'weight': weight,
                    'fs': 1000,
                    'duration': 2.0,
                    'noise_level': 0.0
                }
                
                # Generate the signal
                t, x = SignalGenerator.generate_continuous_signal(signal_type, signal_params)
                
                st.session_state.mixing_signals.append({
                    'name': f"{signal_type}_{len(st.session_state.mixing_signals)+1}",
                    'params': signal_params,
                    't': t,
                    'x': x,
                    'weight': weight
                })
                st.success(f"✅ Added {signal_type} signal")
                st.rerun()
    
    with col2:
        st.markdown("### 📊 Signal List")
        
        if st.session_state.mixing_signals:
            for i, sig in enumerate(st.session_state.mixing_signals):
                with st.container():
                    st.markdown(f"**{sig['name']}**")
                    st.write(f"Type: {sig['params']['type']}")
                    st.write(f"Weight: {sig['weight']:.2f}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        # Update weight
                        new_weight = st.slider(f"Weight", 0.0, 2.0, sig['weight'], 0.1, key=f"weight_{i}")
                        if new_weight != sig['weight']:
                            st.session_state.mixing_signals[i]['weight'] = new_weight
                    
                    with col_b:
                        if st.button("🗑️ Remove", key=f"remove_{i}"):
                            st.session_state.mixing_signals.pop(i)
                            st.rerun()
                    
                    st.divider()
        else:
            st.info("No signals added yet")
        
        # Clear all button
        if st.session_state.mixing_signals and st.button("🗑️ Clear All"):
            st.session_state.mixing_signals = []
            st.rerun()
    
    # Mix and visualize signals
    if len(st.session_state.mixing_signals) >= 1:
        st.markdown("### 🎛️ Mixed Signal Analysis")
        
        # Extract signals and weights
        signals = [sig['x'] for sig in st.session_state.mixing_signals]
        weights = [sig['weight'] for sig in st.session_state.mixing_signals]
        t = st.session_state.mixing_signals[0]['t']  # Use time from first signal
        
        # Mix the signals
        mixed_signal = SignalProcessor.mix_signals(*signals, weights=weights)
        
        # Visualization options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_individual = st.checkbox("Show Individual Signals", value=True)
        with col2:
            show_mixed = st.checkbox("Show Mixed Signal", value=True)
        with col3:
            normalize_mixed = st.checkbox("Normalize Mixed Signal", help="Scale to prevent clipping")
        
        if normalize_mixed:
            max_val = np.max(np.abs(mixed_signal))
            if max_val > 0:
                mixed_signal = mixed_signal / max_val
        
        # Plot signals
        fig = go.Figure()
        
        if show_individual:
            for i, sig in enumerate(st.session_state.mixing_signals):
                fig.add_trace(go.Scatter(
                    x=t[:len(sig['x'])], 
                    y=sig['x'] * sig['weight'],
                    mode='lines',
                    name=f"{sig['name']} (w={sig['weight']:.2f})",
                    opacity=0.7,
                    line=dict(width=1)
                ))
        
        if show_mixed:
            fig.add_trace(go.Scatter(
                x=t[:len(mixed_signal)],
                y=mixed_signal,
                mode='lines',
                name='Mixed Signal',
                line=dict(color='red', width=3)
            ))
        
        fig.update_layout(
            title="Signal Mixing Visualization",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Mixed signal analysis
        col1, col2, col3, col4 = st.columns(4)
        power_metrics = SignalProcessor.calculate_power_energy(mixed_signal)
        
        with col1:
            st.metric("Peak Amplitude", f"{np.max(np.abs(mixed_signal)):.3f}")
        with col2:
            st.metric("RMS Value", f"{power_metrics['rms']:.3f}")
        with col3:
            st.metric("Signal Power", f"{power_metrics['power']:.3f}")
        with col4:
            st.metric("Crest Factor", f"{power_metrics['crest_factor']:.2f}")
        
        # Export mixed signal
        st.markdown("### 💾 Export Mixed Signal")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export CSV"):
                csv_data = SignalExporter.export_to_csv(t[:len(mixed_signal)], mixed_signal)
                st.download_button("📊 Download CSV", csv_data, "mixed_signal.csv", "text/csv")
        
        with col2:
            if st.button("📊 Export MAT"):
                mat_data = SignalExporter.export_to_mat(t[:len(mixed_signal)], mixed_signal)
                st.download_button("📊 Download MAT", mat_data, "mixed_signal.mat", "application/octet-stream")
        
        with col3:
            if st.button("🖼️ Export PNG"):
                png_data = SignalExporter.export_to_png(t[:len(mixed_signal)], mixed_signal, "Continuous-Time", "mixed_signal")
                st.download_button("🖼️ Download PNG", png_data, "mixed_signal.png", "image/png")
        
        # Save mixed signal for further analysis
        if st.button("💾 Save for Analysis"):
            st.session_state.current_signal = (
                t[:len(mixed_signal)], 
                mixed_signal, 
                "Continuous-Time", 
                "mixed_signal", 
                {'fs': 1000, 'duration': 2.0}
            )
            st.success("✅ Mixed signal saved! Switch to 'Generated Signals' mode to analyze.")
        
        # Frequency analysis of mixed signal
        if st.checkbox("Show Frequency Analysis"):
            st.markdown("### 🔍 Mixed Signal Frequency Analysis")
            
            # Compute FFT
            X_mixed = fft(mixed_signal)
            fs = 1000  # Default sampling rate
            freqs = fftfreq(len(mixed_signal), 1/fs)
            
            # Plot frequency spectrum
            fig_freq = go.Figure()
            fig_freq.add_trace(go.Scatter(
                x=freqs[:len(freqs)//2],
                y=np.abs(X_mixed[:len(X_mixed)//2]),
                mode='lines',
                name='Mixed Signal Spectrum',
                line=dict(color='purple', width=2)
            ))
            
            fig_freq.update_layout(
                title="Mixed Signal Frequency Spectrum",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                height=400
            )
            st.plotly_chart(fig_freq, use_container_width=True)
            
            # Dominant frequency
            dominant_idx = np.argmax(np.abs(X_mixed[:len(X_mixed)//2]))
            dominant_freq = freqs[dominant_idx]
            st.info(f"🎯 **Dominant Frequency:** {dominant_freq:.2f} Hz")
    
    else:
        st.info("👆 Add at least one signal to start mixing!")

def show_realtime_interface():
    st.markdown("""
    <div class="feature-card">
        <h2 style="color: #667eea; margin-bottom: 1rem;">⚡ Real-Time Signal Streaming</h2>
        <p style="color: #64748b; margin: 0;">Live signal generation with interactive visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    rt_gen = st.session_state.realtime_generator
    
    if rt_gen.is_running:
        # Enhanced status indicator
        st.markdown("""
        <div class="status-indicator status-success" style="margin: 1rem 0;">
            🟢 Real-time signal is streaming...
        </div>
        """, unsafe_allow_html=True)
        
        # Create stable containers to reduce flickering
        chart_container = st.container()
        metrics_container = st.container()
        
        # Get current buffer data
        t_buffer, x_buffer = rt_gen.get_buffer_data()
        
        if len(x_buffer) > 0:
            # Live signal visualization
            with chart_container:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=t_buffer,
                    y=x_buffer,
                    mode='lines',
                    name='Real-Time Signal',
                    line=dict(color='#667eea', width=2),
                    line_shape='spline'
                ))
                
                fig.update_layout(
                    title={
                        'text': "🌊 Live Signal Stream",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20, 'color': '#334155'}
                    },
                    xaxis_title="Time (s)", 
                    yaxis_title="Amplitude",
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, sans-serif"),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                fig.update_xaxes(gridcolor='rgba(0,0,0,0.1)', gridwidth=1)
                fig.update_yaxes(gridcolor='rgba(0,0,0,0.1)', gridwidth=1)
                # Lock x-axis to a rolling time window for clearer motion
                if len(t_buffer) > 2:
                    tmax = float(t_buffer[-1])
                    tmin = float(t_buffer[0])
                    window = 5.0
                    if tmax - tmin > window:
                        fig.update_xaxes(range=[tmax - window, tmax])
                
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Use session state counter for chart key
                st.plotly_chart(fig, use_container_width=True, key=f"realtime_chart_{st.session_state.update_counter}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced real-time metrics with modern cards
            with metrics_container:
                st.markdown("### 📊 Live Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #667eea; margin: 0;">Current Value</h4>
                        <h2 style="margin: 0.5rem 0 0 0; color: #334155;">{x_buffer[-1]:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    rms_val = np.sqrt(np.mean(np.array(x_buffer)**2))
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #667eea; margin: 0;">RMS Level</h4>
                        <h2 style="margin: 0.5rem 0 0 0; color: #334155;">{rms_val:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    peak_val = np.max(np.abs(x_buffer))
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #667eea; margin: 0;">Peak Level</h4>
                        <h2 style="margin: 0.5rem 0 0 0; color: #334155;">{peak_val:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #667eea; margin: 0;">Buffer Size</h4>
                        <h2 style="margin: 0.5rem 0 0 0; color: #334155;">{len(x_buffer)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Real-time frequency analysis - force display
            st.markdown("### 🔍 Real-Time Frequency Analysis")
            
            if len(x_buffer) > 5:
                # Compute FFT for last N seconds to reflect current content
                x_array = np.array(x_buffer)
                t_array = np.array(t_buffer)
                # Keep a 5s window for frequency estimate
                if len(t_array) > 2:
                    tmax = float(t_array[-1])
                    mask = t_array >= (tmax - 5.0)
                    if mask.any():
                        x_array = x_array[mask]
                        t_array = t_array[mask]
                
                try:
                    X_fft = fft(x_array)
                    dt = np.mean(np.diff(t_array)) if len(t_array) > 1 else 1.0/rt_gen.sample_rate
                    freqs = fftfreq(len(x_array), dt)
                    
                    # Create frequency domain plot
                    fig_freq = go.Figure()
                    fig_freq.add_trace(go.Scatter(
                        x=freqs[:len(freqs)//2],
                        y=np.abs(X_fft[:len(X_fft)//2]),
                        mode='lines',
                        name='Frequency Spectrum',
                        line=dict(color='#667eea', width=2)
                    ))
                    
                    fig_freq.update_layout(
                        title={
                            'text': "🌊 Real-Time Frequency Spectrum",
                            'x': 0.5,
                            'xanchor': 'center',
                            'font': {'size': 18, 'color': '#334155'}
                        },
                        xaxis_title="Frequency (Hz)",
                        yaxis_title="Magnitude",
                        height=350,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    # Use session state counter for chart key
                    st.plotly_chart(fig_freq, use_container_width=True, key=f"rt_freq_chart_{st.session_state.update_counter}")
                    
                    # Frequency domain metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if len(X_fft) > 2:
                            dominant_idx = np.argmax(np.abs(X_fft[:len(X_fft)//2]))
                            dominant_freq = freqs[dominant_idx] if dominant_idx < len(freqs) else 0
                            st.metric("Dominant Frequency", f"{abs(dominant_freq):.2f} Hz")
                        else:
                            st.metric("Dominant Frequency", "N/A")
                    
                    with col2:
                        if len(X_fft) > 2:
                            magnitude = np.abs(X_fft[:len(X_fft)//2])
                            freq_slice = freqs[:len(freqs)//2]
                            if np.sum(magnitude) > 0:
                                spectral_centroid = np.sum(np.abs(freq_slice) * magnitude) / np.sum(magnitude)
                                st.metric("Spectral Centroid", f"{spectral_centroid:.2f} Hz")
                            else:
                                st.metric("Spectral Centroid", "0.00 Hz")
                        else:
                            st.metric("Spectral Centroid", "N/A")
                    
                    with col3:
                        st.metric("Buffer Length", f"{len(x_buffer)} samples")
                        
                except Exception as e:
                    st.error(f"FFT Error: {str(e)}")
                    st.write(f"Buffer info: {len(x_buffer)} samples, time range: {t_array[0]:.3f} to {t_array[-1]:.3f}")
            else:
                st.info(f"Waiting for more data... (have {len(x_buffer)} samples, need >5)")
            
            # Real-time classification
            if len(x_buffer) > 20:
                st.markdown("### 🤖 Real-Time Classification")
                features = SignalClassifier.extract_features(np.array(t_buffer), np.array(x_buffer))
                predicted_type = SignalClassifier.classify_signal_type(features)
                st.info(f"**Detected Signal Type:** {predicted_type}")
        
        # Optimized refresh mechanism to reduce blinking
        if 'last_update' not in st.session_state:
            st.session_state.last_update = time.time()
        if 'update_counter' not in st.session_state:
            st.session_state.update_counter = 0
        
        current_time = time.time()
        # Reduce refresh rate and add frame skipping
        if current_time - st.session_state.last_update > 0.2:  # Update ~5 FPS
            rt_gen.advance()  # Generate as many samples as wall-time requires
            st.session_state.last_update = current_time
            st.session_state.update_counter += 1
            
            # Only rerun every few updates to reduce flickering
            if st.session_state.update_counter % 2 == 0:
                st.rerun()
    else:
        # Enhanced inactive state with modern design
        st.markdown("""
        <div class="feature-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: #64748b; margin-bottom: 1rem;">⏸️ Real-Time Streaming Inactive</h3>
            <p style="color: #94a3b8; margin-bottom: 2rem;">Configure your signal parameters in the sidebar and start streaming</p>
            <div class="status-indicator status-warning" style="display: inline-flex;">
                🎛️ Use the Control Center to begin
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_comparison_interface():
    st.markdown("## 🔄 Signal Comparison: Generated vs Real-Time")
    
    rt_gen = st.session_state.realtime_generator
    
    # Check if we have both signals
    has_generated = st.session_state.current_signal is not None
    has_realtime = rt_gen.is_running and len(rt_gen.buffer) > 0
    
    if has_generated and has_realtime:
        # Get both signals
        t_gen, x_gen, domain, sig_type, params = st.session_state.current_signal
        t_rt, x_rt = rt_gen.get_buffer_data()
        
        # Comparison visualization
        fig = make_subplots(rows=2, cols=2, 
                          subplot_titles=('Generated Signal', 'Real-Time Signal', 
                                        'Frequency Comparison', 'Feature Comparison'))
        
        # Generated signal
        fig.add_trace(go.Scatter(x=t_gen, y=x_gen, mode='lines', name='Generated', 
                               line=dict(color='blue')), row=1, col=1)
        
        # Real-time signal
        fig.add_trace(go.Scatter(x=t_rt, y=x_rt, mode='lines', name='Real-Time', 
                               line=dict(color='red')), row=1, col=2)
        
        # Frequency comparison
        if len(x_gen) > 10 and len(x_rt) > 10:
            # FFT for generated signal
            X_gen = fft(x_gen)
            freqs_gen = fftfreq(len(x_gen), t_gen[1] - t_gen[0])
            
            # FFT for real-time signal  
            X_rt = fft(x_rt)
            freqs_rt = fftfreq(len(x_rt), np.mean(np.diff(t_rt)) if len(t_rt) > 1 else 0.01)
            
            fig.add_trace(go.Scatter(x=freqs_gen[:len(freqs_gen)//2], y=np.abs(X_gen[:len(X_gen)//2]), 
                                   mode='lines', name='Generated FFT', line=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=freqs_rt[:len(freqs_rt)//2], y=np.abs(X_rt[:len(X_rt)//2]), 
                                   mode='lines', name='Real-Time FFT', line=dict(color='red')), row=2, col=1)
        
        # Feature comparison
        features_gen = SignalClassifier.extract_features(t_gen, x_gen)
        features_rt = SignalClassifier.extract_features(np.array(t_rt), np.array(x_rt))
        
        common_features = ['rms', 'peak', 'std', 'zero_crossings']
        feature_names = []
        gen_values = []
        rt_values = []
        
        for feat in common_features:
            if feat in features_gen and feat in features_rt:
                feature_names.append(feat)
                gen_values.append(features_gen[feat])
                rt_values.append(features_rt[feat])
        
        if feature_names:
            x_pos = np.arange(len(feature_names))
            fig.add_trace(go.Bar(x=feature_names, y=gen_values, name='Generated', 
                               marker_color='blue', opacity=0.7), row=2, col=2)
            fig.add_trace(go.Bar(x=feature_names, y=rt_values, name='Real-Time', 
                               marker_color='red', opacity=0.7), row=2, col=2)
        
        fig.update_layout(
            height=800, 
            title="Signal Comparison Dashboard",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        
        # Use container and unique key to prevent flickering
        comparison_container = st.container()
        with comparison_container:
            # Use session state counter for chart key
            st.plotly_chart(fig, use_container_width=True, key=f"comparison_chart_{st.session_state.comp_update_counter}")
        
        # Classification comparison
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Generated Signal Classification")
            pred_gen = SignalClassifier.classify_signal_type(features_gen)
            st.info(f"**Type:** {sig_type.title()}")
            st.info(f"**Predicted:** {pred_gen}")
            
        with col2:
            st.markdown("### Real-Time Signal Classification")
            pred_rt = SignalClassifier.classify_signal_type(features_rt)
            st.info(f"**Predicted:** {pred_rt}")
            
        # Similarity analysis
        st.markdown("### 📊 Similarity Analysis")
        
        # Calculate correlation if signals have similar lengths
        min_len = min(len(x_gen), len(x_rt))
        if min_len > 10:
            correlation = np.corrcoef(x_gen[:min_len], x_rt[:min_len])[0, 1]
            st.metric("Cross-Correlation", f"{correlation:.3f}")
            
            if correlation > 0.8:
                st.success("🟢 Signals are highly similar!")
            elif correlation > 0.5:
                st.warning("🟡 Signals are moderately similar")
            else:
                st.error("🔴 Signals are quite different")
        
        # Optimized refresh mechanism for comparison interface
        if 'comp_last_update' not in st.session_state:
            st.session_state.comp_last_update = time.time()
        if 'comp_update_counter' not in st.session_state:
            st.session_state.comp_update_counter = 0
        
        if rt_gen.is_running:
            current_time = time.time()
            if current_time - st.session_state.comp_last_update > 0.5:  # Update every 500ms
                rt_gen.get_next_sample()
                st.session_state.comp_last_update = current_time
                st.session_state.comp_update_counter += 1
                
                # Only rerun every few updates to reduce flickering
                if st.session_state.comp_update_counter % 4 == 0:  # Skip 3 out of 4 updates
                    st.rerun()
            
    else:
        st.warning("⚠️ Need both generated and real-time signals for comparison")
        st.info("Both signals need to be active for comparison to work.")

def show_microphone_interface():
    st.markdown("## 🎤 Microphone Signal Analysis & Speech Recognition")
    
    mic_capture = st.session_state.mic_capture
    
    if mic_capture.is_recording:
        # Get audio data
        t_audio, x_audio = mic_capture.get_buffer_data()
        
        if len(x_audio) > 10:
            # Create audio visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t_audio,
                y=x_audio,
                mode='lines',
                name='Audio Signal',
                line=dict(color='#e74c3c', width=1)
            ))
            
            # Audio frequency analysis
            if len(x_audio) > 50:
                fft_data = fft(np.array(x_audio))
                freqs = fftfreq(len(x_audio), 1/mic_capture.sample_rate)
                
                fig.update_layout(
                    title="Audio Frequency Spectrum",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Magnitude",
                    height=300
                )
                # Use session state counter for chart key
                st.plotly_chart(fig, use_container_width=True, key=f"mic_chart_{st.session_state.mic_update_counter}")
                
                # Dominant frequency
                dominant_freq_idx = np.argmax(np.abs(fft_data[:len(fft_data)//2]))
                dominant_freq = freqs[dominant_freq_idx]
                st.info(f"**Dominant Frequency:** {dominant_freq:.1f} Hz")
                
            # Audio classification
            if sr is not None and len(x_audio) > 100:
                features = SignalClassifier.extract_features(np.array(t_audio), np.array(x_audio))
                predicted_type = SignalClassifier.classify_signal_type(features)
                
                # Simple audio classification based on frequency content
                if len(x_audio) > 50:
                    fft_data = fft(np.array(x_audio))
                    dominant_freq_idx = np.argmax(np.abs(fft_data[:len(fft_data)//2]))
                    freqs = fftfreq(len(x_audio), 1/mic_capture.sample_rate)
                    dominant_freq = freqs[dominant_freq_idx]
                    
                    if 80 <= dominant_freq <= 300:
                        audio_class = "Speech/Voice"
                    elif 200 <= dominant_freq <= 2000:
                        audio_class = "Music/Tonal"
                    else:
                        audio_class = "General Audio"
                else:
                    audio_class = "General Audio"
                
                st.info(f"**Audio Classification:** {audio_class}")
                st.info(f"**Signal Pattern:** {predicted_type}")
                
            # Speech recognition section
            if sr is not None and len(x_audio) > 1000:
                st.markdown("### 🗣️ Speech Recognition")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🎯 Recognize Speech", help="Convert audio to text"):
                        with st.spinner("Processing speech..."):
                            try:
                                # Get recent audio for recognition
                                recent_audio = x_audio[-8000:] if len(x_audio) > 8000 else x_audio
                                text_result = mic_capture.recognize_speech(recent_audio)
                                
                                if text_result and text_result not in ["Could not understand audio", "Insufficient audio data"]:
                                    st.success(f"**Recognized Text:** {text_result}")
                                else:
                                    st.warning("🔇 Could not recognize speech clearly")
                            except Exception as e:
                                st.error(f"Speech recognition error: {str(e)}")
                
                with col2:
                    # Audio level indicator
                    if len(x_audio) > 0:
                        audio_level = np.sqrt(np.mean(np.array(x_audio[-1000:])**2)) if len(x_audio) >= 1000 else np.sqrt(np.mean(np.array(x_audio)**2))
                        if audio_level > 0.01:
                            st.markdown(f"""
                            <div class="status-indicator status-success" style="margin: 0.5rem 0;">
                                🔊 Audio level: {audio_level:.3f}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="status-indicator status-warning" style="margin: 0.5rem 0;">
                                🔇 Audio level too low for recognition
                            </div>
                            """, unsafe_allow_html=True)
        
        # Optimized refresh mechanism for microphone interface
        if 'mic_last_update' not in st.session_state:
            st.session_state.mic_last_update = time.time()
        if 'mic_update_counter' not in st.session_state:
            st.session_state.mic_update_counter = 0
        
        if mic_capture.is_recording:
            current_time = time.time()
            if current_time - st.session_state.mic_last_update > 0.2:  # Update every 200ms
                mic_capture.get_audio_chunk()
                st.session_state.mic_last_update = current_time
                st.session_state.mic_update_counter += 1
            
            # Only rerun every few updates to reduce flickering
            if st.session_state.mic_update_counter % 5 == 0:
                try:
                    st.rerun()
                except Exception as e:
                    pass  # Silently handle refresh errors
        
    else:
        st.warning("🎙️ Microphone is not recording. Use the sidebar to start recording.")
        st.info("Click 'Start Recording' in the sidebar to begin capturing audio from your microphone.")
        
        # Show microphone setup instructions
        st.markdown("### 🔧 Microphone Setup")
        st.markdown("""
        **Requirements:**
        - Working microphone connected to your computer
        - Browser permissions for microphone access (if running in browser)
        - PyAudio library installed for audio capture
        
        **Features:**
        - Real-time audio waveform visualization
        - Live speech recognition using Google Speech API
        - Audio signal feature extraction and analysis
        - Frequency spectrum analysis
        - Audio signal classification
        """)

def show_signal_visualization(t_or_n, x, domain, sig_type):
    st.markdown(f"## 📈 {domain} Signal Visualization")
    st.markdown(f"**Signal Type:** {sig_type.title()}")
    
    # Enhanced signal metrics with power/energy analysis
    fs = len(t_or_n) / (t_or_n[-1] - t_or_n[0]) if len(t_or_n) > 1 and domain == "Continuous-Time" else 1000
    power_metrics = SignalProcessor.calculate_power_energy(x, fs)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Peak Amplitude", f"{np.max(np.abs(x)):.3f}")
    with col2:
        st.metric("RMS Value", f"{power_metrics['rms']:.3f}")
    with col3:
        st.metric("Signal Power", f"{power_metrics['power']:.3f}")
    with col4:
        st.metric("Signal Energy", f"{power_metrics['energy']:.3f}")
    with col5:
        st.metric("Crest Factor", f"{power_metrics['crest_factor']:.2f}")
    
    # Signal processing controls
    st.markdown("### 🔧 Signal Processing")
    col1, col2, col3 = st.columns(3)
    
    # Initialize processing variables with defaults
    apply_filter = False
    add_noise = False
    filter_type = "lowpass"
    cutoff = int(fs/10)
    noise_level = 0.1
    
    with col1:
        apply_filter = st.checkbox("Apply Filter")
        if apply_filter:
            filter_type = st.selectbox("Filter Type", ["lowpass", "highpass", "bandpass", "bandstop"])
            if filter_type in ["lowpass", "highpass"]:
                cutoff = st.slider("Cutoff Frequency (Hz)", 1, int(fs/2), int(fs/10))
            else:
                low_cutoff = st.slider("Low Cutoff (Hz)", 1, int(fs/4), int(fs/20))
                high_cutoff = st.slider("High Cutoff (Hz)", int(fs/4), int(fs/2), int(fs/5))
                cutoff = (low_cutoff, high_cutoff)
    
    with col2:
        add_noise = st.checkbox("Add Noise")
        if add_noise:
            noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1, 0.05)
    
    # Process signal based on user selections FIRST
    processed_x = x.copy()
    
    if apply_filter:
        try:
            processed_x = SignalProcessor.apply_filter(processed_x, fs, filter_type, cutoff)
            st.success(f"✅ Applied {filter_type} filter")
        except Exception as e:
            st.error(f"Filter error: {str(e)}")
    
    if add_noise:
        processed_x += np.random.normal(0, noise_level, len(processed_x))
        st.success(f"✅ Added noise (level: {noise_level})")
    
    with col3:
        # Export options
        st.markdown("**📁 Export Options**")
        export_format = st.selectbox("Format", ["CSV", "MAT", "PNG"], key="export_format")
        
        # Determine payload to export
        export_x = processed_x if (apply_filter or add_noise) else x
        export_label = "processed" if (apply_filter or add_noise) else "original"
        st.info(f"Will export: **{export_label}** signal  |  Data points: {len(export_x)}")

        data = None
        file_name = None
        mime = None
        if export_format == "CSV":
            data = SignalExporter.export_to_csv(t_or_n, export_x)
            file_name = f"signal_{sig_type}_{export_label}.csv"
            mime = "text/csv"
        elif export_format == "MAT":
            data = SignalExporter.export_to_mat(t_or_n, export_x)
            file_name = f"signal_{sig_type}_{export_label}.mat"
            mime = "application/octet-stream"
        elif export_format == "PNG":
            png_processed_x = processed_x if (apply_filter or add_noise) else None
            data = SignalExporter.export_to_png(t_or_n, x, domain, sig_type, png_processed_x)
            file_name = f"signal_{sig_type}_{export_label}.png"
            mime = "image/png"

        st.download_button(label="💾 Export Data", data=data, file_name=file_name, mime=mime, key=f"download_{export_format}")
    
    # Plot original and processed signals
    fig = go.Figure()
    
    if domain == "Continuous-Time":
        fig.add_trace(go.Scatter(
            x=t_or_n,
            y=x,
            mode='lines',
            name='Original Signal',
            line=dict(color='blue', width=2),
            line_shape='spline'
        ))
        if apply_filter or add_noise:
            fig.add_trace(go.Scatter(
                x=t_or_n,
                y=processed_x,
                mode='lines',
                name='Processed Signal',
                line=dict(color='red', width=2, dash='dash'),
                line_shape='spline'
            ))
        fig.update_xaxes(title_text="Time (s)")
    else:
        fig.add_trace(go.Scatter(x=t_or_n, y=np.zeros_like(t_or_n), mode='lines', 
                                line=dict(color='blue', width=1), showlegend=False))
        fig.add_trace(go.Scatter(x=t_or_n, y=x, mode='markers', name='Original Signal', 
                                marker=dict(color='blue', size=8, symbol='circle')))
        
        # Add vertical lines for stem plot
        for i in range(len(t_or_n)):
            fig.add_shape(type="line", x0=t_or_n[i], y0=0, x1=t_or_n[i], y1=x[i],
                         line=dict(color="blue", width=2))
        
        if apply_filter or add_noise:
            fig.add_trace(go.Scatter(x=t_or_n, y=processed_x, mode='markers', name='Processed Signal', 
                                    marker=dict(color='red', size=6, symbol='circle')))
            # Add vertical lines for processed signal
            for i in range(len(t_or_n)):
                fig.add_shape(type="line", x0=t_or_n[i], y0=0, x1=t_or_n[i], y1=processed_x[i],
                             line=dict(color="red", width=2, dash="dash"))
        
        fig.update_xaxes(title_text="Sample Index (n)")
    
    fig.update_yaxes(title_text="Amplitude")
    fig.update_layout(title=f"{domain} Signal: {sig_type.title()}", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Store processed signal for other analyses
    if 'processed_signal' not in st.session_state:
        st.session_state.processed_signal = {}
    st.session_state.processed_signal = {
        't_or_n': t_or_n,
        'x': processed_x,
        'fs': fs,
        'domain': domain,
        'sig_type': sig_type
    }

def show_frequency_analysis(t_or_n, x, domain):
    st.markdown("## 🔍 Frequency Domain Analysis")
    
    if len(x) < 4:
        st.warning("Signal too short for meaningful frequency analysis")
        return
    
    # Get processed signal if available
    if 'processed_signal' in st.session_state and st.session_state.processed_signal:
        processed_data = st.session_state.processed_signal
        if len(processed_data['x']) == len(x):
            x = processed_data['x']  # Use processed signal
            fs = processed_data['fs']
        else:
            fs = len(t_or_n) / (t_or_n[-1] - t_or_n[0]) if len(t_or_n) > 1 and domain == "Continuous-Time" else 1000
    else:
        fs = len(t_or_n) / (t_or_n[-1] - t_or_n[0]) if len(t_or_n) > 1 and domain == "Continuous-Time" else 1000
    
    # Compute FFT
    X = fft(x)
    
    # Add spectrogram option
    col1, col2 = st.columns(2)
    with col1:
        show_spectrogram = st.checkbox("Show Spectrogram", help="2D time-frequency representation")
    with col2:
        show_3d_spectrogram = st.checkbox("Show 3D Spectrogram", help="3D surface plot of spectrogram")
    
    if show_spectrogram or show_3d_spectrogram:
        try:
            f_spec, t_spec, Sxx = SignalProcessor.generate_spectrogram(x, fs)
            
            if show_spectrogram:
                st.markdown("### 📊 Spectrogram (2D)")
                fig_spec = go.Figure(data=go.Heatmap(
                    z=10*np.log10(Sxx + 1e-10),  # Convert to dB, avoid log(0)
                    x=t_spec,
                    y=f_spec,
                    colorscale='Viridis',
                    colorbar=dict(title="Power (dB)")
                ))
                fig_spec.update_layout(
                    title="Spectrogram",
                    xaxis_title="Time (s)",
                    yaxis_title="Frequency (Hz)",
                    height=400
                )
                st.plotly_chart(fig_spec, use_container_width=True)
            
            if show_3d_spectrogram:
                st.markdown("### 🌊 3D Spectrogram")
                fig_3d = go.Figure(data=[go.Surface(
                    z=10*np.log10(Sxx + 1e-10),
                    x=t_spec,
                    y=f_spec,
                    colorscale='Viridis',
                    colorbar=dict(title="Power (dB)")
                )])
                fig_3d.update_layout(
                    title="3D Spectrogram",
                    scene=dict(
                        xaxis_title="Time (s)",
                        yaxis_title="Frequency (Hz)",
                        zaxis_title="Power (dB)"
                    ),
                    height=600
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                
        except Exception as e:
            st.error(f"Spectrogram generation error: {str(e)}")
    
    # Regular FFT analysis
    if domain == "Continuous-Time":
        dt = t_or_n[1] - t_or_n[0]
        freqs = fftfreq(len(x), dt)
    else:
        freqs = fftfreq(len(x), 1.0)  # Normalized frequency for discrete signals
    
    # Plot magnitude spectrum
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Magnitude Spectrum', 'Phase Spectrum'))
    
    # Magnitude
    magnitude = np.abs(X)
    fig.add_trace(go.Scatter(x=freqs[:len(freqs)//2], y=magnitude[:len(magnitude)//2], 
                            mode='lines', name='Magnitude'), row=1, col=1)
    
    # Phase
    phase = np.angle(X)
    fig.add_trace(go.Scatter(x=freqs[:len(freqs)//2], y=phase[:len(phase)//2], 
                            mode='lines', name='Phase', line=dict(color='red')), row=2, col=1)
    
    if domain == "Continuous-Time":
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    else:
        fig.update_xaxes(title_text="Normalized Frequency (cycles/sample)", row=2, col=1)
    
    fig.update_yaxes(title_text="Magnitude", row=1, col=1)
    fig.update_yaxes(title_text="Phase (radians)", row=2, col=1)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Frequency domain metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        dominant_idx = np.argmax(magnitude[:len(magnitude)//2])
        dominant_freq = freqs[dominant_idx]
        st.metric("Dominant Frequency", f"{dominant_freq:.3f}")
    with col2:
        spectral_energy = np.sum(magnitude**2)
        st.metric("Spectral Energy", f"{spectral_energy:.1f}")
    with col3:
        bandwidth = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
        st.metric("Spectral Centroid", f"{bandwidth:.3f}")

def show_signal_properties(t_or_n, x, domain):
    st.markdown("## 📊 Signal Properties & Statistics")
    
    # Automatic Signal Info Panel
    st.markdown("### 🤖 Automatic Signal Information")
    
    # Calculate comprehensive signal info
    fs = len(t_or_n) / (t_or_n[-1] - t_or_n[0]) if len(t_or_n) > 1 and domain == "Continuous-Time" else 1000
    duration = (t_or_n[-1] - t_or_n[0]) if len(t_or_n) > 1 and domain == "Continuous-Time" else len(t_or_n)
    
    # Power and energy metrics
    power_metrics = SignalProcessor.calculate_power_energy(x, fs)
    
    # Frequency analysis for dominant frequency
    X = fft(x)
    if domain == "Continuous-Time":
        freqs = fftfreq(len(x), 1/fs)
    else:
        freqs = fftfreq(len(x), 1.0)
    
    dominant_idx = np.argmax(np.abs(X[:len(X)//2]))
    dominant_freq = abs(freqs[dominant_idx])
    
    # Auto-detect signal characteristics
    zero_crossings = len(np.where(np.diff(np.signbit(x)))[0])
    peak_to_peak = np.max(x) - np.min(x)
    
    # Signal type classification
    if zero_crossings == 0:
        if np.std(x) < 0.01:
            auto_signal_type = "DC/Constant"
        else:
            auto_signal_type = "Step/Ramp"
    elif zero_crossings < len(x) * 0.1:
        auto_signal_type = "Low Frequency/Exponential"
    elif peak_to_peak / power_metrics['rms'] > 4:
        auto_signal_type = "Square/Pulse"
    elif 0.5 < zero_crossings / len(x) < 2.0:
        auto_signal_type = "Sinusoidal"
    else:
        auto_signal_type = "Complex/Noise"
    
    # Display auto-detected info in organized panels
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: white;">📋 Basic Info</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Signal Type", auto_signal_type)
        st.metric("Duration", f"{duration:.3f} {'s' if domain == 'Continuous-Time' else 'samples'}")
        st.metric("Sampling Rate", f"{fs:.0f} Hz" if domain == "Continuous-Time" else "N/A")
        st.metric("Data Points", f"{len(x)}")
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: white;">⚡ Power & Energy</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Signal Power", f"{power_metrics['power']:.6f}")
        st.metric("Signal Energy", f"{power_metrics['energy']:.6f}")
        st.metric("RMS Value", f"{power_metrics['rms']:.6f}")
        st.metric("Crest Factor", f"{power_metrics['crest_factor']:.3f}")
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: white;">🔍 Frequency Info</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Dominant Frequency", f"{dominant_freq:.3f} Hz")
        st.metric("Zero Crossings", f"{zero_crossings}")
        st.metric("Peak-to-Peak", f"{peak_to_peak:.6f}")
        st.metric("Dynamic Range", f"{20*np.log10(np.max(np.abs(x))/np.min(np.abs(x[x!=0]))) if len(x[x!=0]) > 0 else 0:.1f} dB")
    
    # Detailed statistical analysis
    st.markdown("### 📈 Statistical Analysis")
    
    # Extract features
    is_discrete = (domain == "Discrete-Time")
    features = SignalClassifier.extract_features(t_or_n, x, is_discrete)
    
    # Display features in a nice format
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Time Domain Properties")
        st.markdown(f'<div class="signal-card">', unsafe_allow_html=True)
        for key, value in list(features.items())[:8]:
            if isinstance(value, (int, float)):
                st.write(f"**{key.replace('_', ' ').title()}:** {value:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Frequency Domain Properties")
        st.markdown(f'<div class="signal-card">', unsafe_allow_html=True)
        freq_features = {k: v for k, v in features.items() if 'freq' in k or 'spectral' in k}
        if freq_features:
            for key, value in freq_features.items():
                if isinstance(value, (int, float)):
                    st.write(f"**{key.replace('_', ' ').title()}:** {value:.4f}")
        else:
            st.write("No frequency domain features available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature visualization
    feature_names = list(features.keys())[:10]  # Show first 10 features
    feature_values = [features[name] for name in feature_names]
    
    fig = go.Figure(data=go.Bar(x=feature_names, y=feature_values))
    fig.update_layout(title="Signal Features", xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_classification(t_or_n, x, domain, actual_type):
    st.markdown("## 🤖 Enhanced Signal Classification")
    
    # Extract features and classify
    is_discrete = (domain == "Discrete-Time")
    features = SignalClassifier.extract_features(t_or_n, x, is_discrete)
    predicted_type = SignalClassifier.classify_signal_type(features)
    
    # Calculate core metrics
    fs = 1000  # Assume 1kHz sampling rate if not provided
    if len(t_or_n) > 1:
        fs = 1.0 / (t_or_n[1] - t_or_n[0]) if not is_discrete else 1.0
    
    # Advanced signal analysis
    thd_info = SignalProcessor.calculate_harmonic_distortion(x, fs) if len(x) > 50 else None
    snr = SignalProcessor.calculate_snr(x)
    
    # Create modern dashboard layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Signal Identity Card
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h3 style="color: white; margin: 0;">📊 Signal Identity Card</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Main classification card
        st.markdown(f'<div class="signal-card" style="padding: 1.5rem; border-radius: 10px;">', unsafe_allow_html=True)
        
        # Basic Info
        if isinstance(actual_type, str):
            actual_type = [actual_type]  # Convert string to list for consistency
        if isinstance(predicted_type, str):
            predicted_type = [predicted_type]  # Convert string to list for consistency
            
        # Display actual type
        st.write("**🎯 Actual Type:**", ", ".join(t.title() for t in actual_type))
        
        # Display predicted types with categorization
        st.write("**🎯 Predicted Classifications:**")
        categories = {
            "Signal Type": lambda x: not any(k in x.lower() for k in ["symmetric", "random", "deterministic", "mixed", "energy", "power"]),
            "Symmetry": lambda x: "symmetric" in x.lower(),
            "Nature": lambda x: any(k in x.lower() for k in ["random", "deterministic", "mixed"]),
            "Energy/Power": lambda x: any(k in x.lower() for k in ["energy", "power"])
        }
        
        for category, filter_func in categories.items():
            matching_types = [t for t in predicted_type if filter_func(t)]
            if matching_types:
                st.write(f"  • {category}: {', '.join(matching_types)}")
        
        # Match indicator - check if any predicted type matches any actual type
        any_match = any(
            any(p.lower() in a.lower() or a.lower() in p.lower() 
                for a in actual_type)
            for p in predicted_type
        )
        match_status = "✅" if any_match else "❌"
        st.write("**🎯 Classification Match:**", match_status)
        
        # Symmetry Analysis - look for symmetry in predicted types
        symmetry_type = next((t for t in predicted_type if "symmetric" in t.lower()), "Unknown")
        symmetry_color = {
            "Even Symmetric": "🟢",
            "Odd Symmetric": "🟡",
            "Non-symmetric": "⚪"
        }.get(symmetry_type, "⚪")
        st.write(f"**🔄 Symmetry Type:** {symmetry_color} {features['symmetry']}")
        
        # Nature Analysis
        nature_color = "🟢" if features['nature'] == "Deterministic" else "🟡"
        st.write(f"**🎲 Signal Nature:** {nature_color} {features['nature']}")
        
        # Energy/Power Analysis
        energy_color = {
            "Energy Signal": "🟢",
            "Power Signal": "🟡",
            "Undefined": "⚪"
        }.get(features['energy_power'], "⚪")
        st.write(f"**⚡ Signal Class:** {energy_color} {features['energy_power']}")
        
        # Elementary Signal Detection
        elementary_color = "🟢" if features['elementary_type'] != "Complex/Other" else "⚪"
        st.write(f"**📐 Elementary Form:** {elementary_color} {features['elementary_type']}")
        
        # Detailed Analysis Expander
        with st.expander("📈 Detailed Analysis"):
            # Symmetry metrics
            st.markdown("#### Symmetry Analysis")
            x_rev = x[::-1]
            even_error = np.mean(np.abs(x - x_rev))
            odd_error = np.mean(np.abs(x + x_rev))
            st.write(f"Even Symmetry Error: {even_error:.6f}")
            st.write(f"Odd Symmetry Error: {odd_error:.6f}")
            
            # Deterministic analysis
            st.markdown("#### Deterministic vs Random")
            fft_x = np.abs(fft(x))
            peak_ratio = np.max(fft_x) / np.mean(fft_x)
            st.write(f"Predictability Score: {1.0 - (np.std(np.diff(x))/(np.std(x) + 1e-12)):.3f}")
            st.write(f"Spectral Peak Ratio: {peak_ratio:.3f}")
            
            # Energy and Power
            st.markdown("#### Energy & Power Metrics")
            if is_discrete:
                energy = np.sum(np.abs(x)**2)
                power = energy / len(x)
            else:
                dt = t_or_n[1] - t_or_n[0]
                energy = np.sum(np.abs(x)**2) * dt
                power = energy / (t_or_n[-1] - t_or_n[0])
            st.write(f"Total Energy: {energy:.3f}")
            st.write(f"Average Power: {power:.3f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Visualization Summary
        st.markdown("""
        <div style="background: linear-gradient(135deg, #134e5e 0%, #71b280 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h3 style="color: white; margin: 0;">🎨 Visual Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Signal Characteristics Plot
        fig = go.Figure()
        
        # Plot original signal
        fig.add_trace(go.Scatter(
            x=t_or_n,
            y=x,
            name='Signal',
            line=dict(color='#00ff00', width=2)
        ))
        
        # Plot reversed signal for symmetry comparison
        fig.add_trace(go.Scatter(
            x=t_or_n,
            y=x[::-1],
            name='Reversed',
            line=dict(color='#ff0000', width=2, dash='dash')
        ))
        
        # Layout
        fig.update_layout(
            title="Signal Symmetry Visualization",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            showlegend=True
        )
        
        # Configure axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_text="Time/Sample")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_text="Amplitude")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Metrics
        st.markdown(f'<div class="signal-card">', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric(
                "Signal Quality",
                f"{snr:.1f} dB",
                delta="Good" if snr > 20 else ("Fair" if snr > 10 else "Poor")
            )
            st.metric(
                "Spectral Purity",
                f"{features['spectral_flatness']:.3f}",
                delta="Clean" if features['spectral_flatness'] < 0.3 else "Noisy"
            )
        
        with col4:
            if thd_info:
                st.metric(
                    "Harmonic Distortion",
                    f"{thd_info['thd']:.3%}",
                    delta="Low" if thd_info['thd'] < 0.1 else "High"
                )
                st.metric(
                    "Fundamental Freq",
                    f"{thd_info['fundamental_freq']:.1f} Hz"
                )
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display predicted types by category
        st.write("**Predicted Classifications:**")
        categories = {
            "Basic Type": lambda x: not any(k in x.lower() for k in ["symmetric", "random", "deterministic", "mixed", "energy", "power"]),
            "Symmetry": lambda x: "symmetric" in x.lower(),
            "Nature": lambda x: any(k in x.lower() for k in ["random", "deterministic", "mixed"]),
            "Energy/Power": lambda x: any(k in x.lower() for k in ["energy", "power"])
        }
        
        for category, filter_func in categories.items():
            matching_types = [t for t in predicted_type if filter_func(t)]
            if matching_types:
                st.write(f"  • {category}: {', '.join(matching_types)}")
        
        # Enhanced accuracy check
        if isinstance(actual_type, str):
            actual_type = [actual_type]
            
        matches = []
        for pred in predicted_type:
            for actual in actual_type:
                if (pred.lower() in actual.lower() or 
                    actual.lower() in pred.lower()):
                    matches.append(pred)
                    break
        
        accuracy_indicator = "✅" if matches else "❌"
        st.write(f"**Classification Match:** {accuracy_indicator}")
        if matches:
            st.write("**Matching Types:**", ", ".join(matches))
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Key Classification Features")
        st.markdown(f'<div class="signal-card">', unsafe_allow_html=True)
        
        # Show most important features for classification
        important_features = ['zero_crossings', 'peak_to_peak', 'rms', 'std', 'skewness', 'kurtosis']
        for feat in important_features:
            if feat in features:
                st.write(f"**{feat.replace('_', ' ').title()}:** {features[feat]:.4f}")
        
        if 'dominant_freq' in features:
            st.write(f"**Dominant Frequency:** {features['dominant_freq']:.4f} Hz")
        
        if 'spectral_flatness' in features:
            st.write(f"**Spectral Flatness:** {features['spectral_flatness']:.4f}")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced metrics
    st.markdown("### Advanced Signal Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="signal-card">', unsafe_allow_html=True)
        st.markdown("#### Frequency Domain Metrics")
        
        if 'spectral_centroid' in features:
            st.write(f"**Spectral Centroid:** {features['spectral_centroid']:.4f} Hz")
        
        if 'spectral_bandwidth' in features:
            st.write(f"**Spectral Bandwidth:** {features['spectral_bandwidth']:.4f} Hz")
            
        if 'spectral_rolloff' in features:
            st.write(f"**Spectral Roll-off:** {features['spectral_rolloff']:.4f} Hz")
            
        if thd_info:
            st.write(f"**Total Harmonic Distortion:** {thd_info['thd']:.4f}")
            st.write(f"**Fundamental Frequency:** {thd_info['fundamental_freq']:.4f} Hz")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown(f'<div class="signal-card">', unsafe_allow_html=True)
        st.markdown("#### Signal Quality Metrics")
        
        st.write(f"**Signal-to-Noise Ratio (SNR):** {snr:.2f} dB")
        
        # Get power and energy metrics
        power_metrics = SignalProcessor.calculate_power_energy(x, fs)
        st.write(f"**Signal Power:** {power_metrics['power']:.6f}")
        st.write(f"**Signal Energy:** {power_metrics['energy']:.6f}")
        st.write(f"**Crest Factor:** {power_metrics['crest_factor']:.4f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Classification explanation
    st.markdown("### Classification Logic")
    st.info("""
    **Enhanced Rule-Based Classification:**
    - **DC/Constant**: No zero crossings, low variation
    - **Step/Ramp**: No/few zero crossings, higher variation  
    - **Exponential Decay**: Few zero crossings, low spectral flatness
    - **Low Frequency**: Few zero crossings, higher spectral content
    - **Noise/Random**: High spectral flatness (>0.5)
    - **Square Wave**: High peak-to-peak ratio, positive kurtosis
    - **Pulse/Sawtooth**: High peak-to-peak ratio, non-positive kurtosis
    - **Sinusoidal**: Moderate kurtosis (0.8-2.0), low skewness
    - **Triangle/Modulated**: Other periodic signals
    - **Complex/Noise**: Irregular patterns
    """)

if __name__ == "__main__":
    main()
