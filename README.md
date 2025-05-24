# EMG-Signal-Generation-using-Wasserstein-GAN-WGAN-
## 🧠 EMG Signal Generation using Wasserstein GAN (WGAN)

This project implements a **Wasserstein Generative Adversarial Network (WGAN)** to generate synthetic EMG (Electromyography) signals based on real data.

### 📦 Description

The code trains a WGAN consisting of:
- A **Generator** that learns to produce fake EMG signals from random noise.
- A **Discriminator** (Critic) that evaluates the realism of signals.
- **Wasserstein loss** and **weight clipping** for stable training.

### How It Works

1. **Load and Preprocess Data**
   - Loads EMG signal from `emg_S1_Iter1_Joint_El_Rest_pos1.csv`
   - Normalizes values to the range [-1, 1]
   - Converts it into a PyTorch tensor

2. **Initialize GAN Components**
   - Uses a latent space (`latent_dim = 1000`) to generate 1D signals of length 1000
   - Uses external `Generator` and `Discriminator` modules
   - RMSprop optimizer is applied for both generator and critic

3. **Training Loop**
   - For each epoch:
     - Updates the critic `n_critic` times to improve its discrimination
     - Updates the generator to fool the critic
     - Tracks loss values for both networks
     - Applies weight clipping to the critic after each update

4. **Signal Generation**
   - After training, the generator produces a synthetic EMG signal
   - The code visualizes:
     - The generated signal
     - A side-by-side comparison with the real signal
     - Generator and critic loss over epochs

### Output Plots
- Generated EMG signal waveform
- Real vs. generated signal comparison
- Loss trends for generator and critic

### Requirements
- `emg_S1_Iter1_Joint_El_Rest_pos1.csv`: Real EMG signal data
- `generator.py`: PyTorch implementation of the Generator
- `Discriminator.py`: PyTorch implementation of the Critic
 
