import numpy as np
import scipy.signal as ss
import numpy as np
from ssspy.bss.fdica import NaturalGradFDICA
from ssspy.bss.ica import NaturalGradICA
import pydub
import io
import soundfile as sf


# Calculates Cartesian coordinates for Eigenmike em32 from spherical angles
# provided in the documentation table (Theta, Phi in degrees).
# Angles: Theta (polar, from +Z), Phi (azimuth, from +X) in degrees.
# Radius: 4.2 cm = 0.042 m.
def get_eigenmike_em32_coords_from_table():
    radius_m = 0.042 

    # Angles from the table (degrees) - Mic 1 to 32
    theta_deg = np.array([
        69, 90, 111, 90, 32, 55, 90, 125, 148, 125, 90, 55, 21, 58, 121, 159,
        69, 90, 111, 90, 32, 55, 90, 125, 148, 125, 90, 55, 21, 58, 122, 159
    ])
    phi_deg = np.array([
         0,  32,   0, 328,   0,  45,  69,  45,   0, 315, 291, 315,  91,  90,  90,  89,
       180, 212, 180, 148, 180, 225, 249, 225, 180, 135, 111, 135, 269, 270, 270, 271
    ])

    # Convert degrees to radians
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)

    # Spherical to Cartesian conversion
    # x = r * sin(theta) * cos(phi)
    # y = r * sin(theta) * sin(phi)
    # z = r * cos(theta)
    x = radius_m * np.sin(theta_rad) * np.cos(phi_rad)
    y = radius_m * np.sin(theta_rad) * np.sin(phi_rad)
    z = radius_m * np.cos(theta_rad)

    # Stack into (3, 32) array
    coords_xyz = np.vstack((x, y, z))

    # Small correction for potential floating point inaccuracies at poles/zeros
    coords_xyz[np.abs(coords_xyz) < 1e-15] = 0.0

    return coords_xyz


# with help of ssspy library we implement Frequency Domain Independent Component Analysis (FDICA)
# and normal Time Domain Independent Component Analysis (ICA)
# combined one after the other these algorithms give the best results
# this approach is based on scientific paper that proposed this method - Multi Stage Independent Component Analysis (MSICA)

# to make this technique even more robust and reliable, we follow another paper published by the same authors
# that proposes to apply whitening before ICA and then dewhiten the data after

# whitening means that we transform the data so that the covariance matrix is the identity matrix
# meaning that the data is uncorrelated and has unit variance

# hyperparameters for these models are chosen by trial and error (not enough time and computational resources)
# but in future a grid search should be done to find the best hyperparameters
# which should make the results even better


def fdica(audio, n_fft=4096, n_iter=500):

    def contrast_fn(y):
        return 2 * np.abs(y)

    def score_fn(y):
        denom = np.maximum(np.abs(y), 1e-10)
        return y / denom

    fdica = NaturalGradFDICA(
        step_size=1e-1,
        contrast_fn=contrast_fn,
        score_fn=score_fn,
        is_holonomic=True,
    )

    hop_length = n_fft // 2

    _, _, spectrogram_mix = ss.stft(
        audio, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length)

    spectrogram_est = fdica(spectrogram_mix, n_iter)
    _, waveform_est = ss.istft(
        spectrogram_est, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length)

    return waveform_est


def ica(audio):

    def contrast_fn(x):
        return np.log(1 + np.exp(x))

    def score_fn(x):
        return 1 / (1 + np.exp(-x))

    ica = NaturalGradICA(
        contrast_fn=contrast_fn, score_fn=score_fn, is_holonomic=True
    )

    waveform_est = ica(audio, n_iter=500)
    return waveform_est

def prewhiten(audio):
    # Compute the covariance matrix
    cov_matrix = np.cov(audio)
    
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Compute the whitening transformation matrix
    whitening_matrix = np.diag(1.0 / np.sqrt(eigenvalues)).dot(eigenvectors.T)
    
    # Whitening the data
    whitened_data = np.dot(whitening_matrix, audio)
    
    return whitened_data

def dewhitening(whitened_data, original_cov_matrix):
    # Calculate the eigenvalues and eigenvectors of the original covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(original_cov_matrix)
    
    # Compute the dewhitening transformation matrix
    dewhitening_matrix = np.diag(np.sqrt(eigenvalues)).dot(eigenvectors)
    
    # Dewhitening the data
    dewhitened_data = np.dot(dewhitening_matrix, whitened_data)
    
    return dewhitened_data


# helper function to convert numpy array to audio segment
def numpy_to_audio_segment(numpy_array: np.ndarray, sample_rate: int, gain: float = -1.0) -> pydub.AudioSegment:
    # convert from double to float
    if numpy_array.dtype != np.float32:
         numpy_array = numpy_array.astype(np.float32)

    # option to increase or decrease the gain of the audio segment
    if gain != -1.0:
        numpy_array = numpy_array * gain

    buffer = io.BytesIO()
    sf.write(buffer, numpy_array, sample_rate, format='WAV')

    buffer.seek(0)

    audio_segment = pydub.AudioSegment.from_wav(buffer)

    return audio_segment