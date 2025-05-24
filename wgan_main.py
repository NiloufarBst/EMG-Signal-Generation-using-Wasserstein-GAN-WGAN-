import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generator import Generator
from Discriminator import Discriminator


file_path = 'emg_S1_Iter1_Joint_El_Rest_pos1.csv'
data = pd.read_csv(file_path, sep=',')
data_p1 = data.iloc[2:10002, 0]
data_p1_normalized = 2 * (data_p1 - np.min(data_p1)) / (np.max(data_p1) - np.min(data_p1)) - 1
real_data_tensor = torch.tensor(data_p1_normalized.values, dtype=torch.float32)

#Param
latent_dim = 1000
signal_length = 1000
batch_size = 128
epochs = 1000
n_critic = 5  #critic updates per generator update
weight_clip = 0.01

generator = Generator()
critic = Discriminator()

optimizer_G = optim.RMSprop(generator.parameters(), lr=0.0005)
optimizer_C = optim.RMSprop(critic.parameters(), lr=0.0005)

def get_real_data(batch_size):
    idx = np.random.randint(0, real_data_tensor.shape[0] - signal_length, batch_size)
    return torch.stack([real_data_tensor[i:i + signal_length] for i in idx])

def generate_noise(batch_size, latent_dim):
    return torch.randn(batch_size, latent_dim)

G_losses = []
C_losses = []

#training
for epoch in range(epochs):
    epoch_G_loss = 0
    epoch_C_loss = 0

    #Training critic
    for _ in range(n_critic):
        critic.zero_grad()
        real_data = get_real_data(batch_size)
        fake_data = generator(generate_noise(batch_size, latent_dim)).detach()

        #Wloss for critic
        loss_real = torch.mean(critic(real_data))
        loss_fake = torch.mean(critic(fake_data))
        c_loss = -(loss_real - loss_fake)  # Negate to minimize
        c_loss.backward()
        optimizer_C.step()

        #Weight clipping
        for p in critic.parameters():
            p.data.clamp_(-weight_clip, weight_clip)

        epoch_C_loss += c_loss.item()

    #training generator
    generator.zero_grad()
    noise = generate_noise(batch_size, latent_dim)
    fake_data = generator(noise)

    #Wloss for generator
    g_loss = -torch.mean(critic(fake_data))
    g_loss.backward()
    optimizer_G.step()

    epoch_G_loss += g_loss.item()
    epoch_C_loss /= n_critic
    G_losses.append(epoch_G_loss)
    C_losses.append(epoch_C_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] Critic Loss: {epoch_C_loss:.4f}, Generator Loss: {epoch_G_loss:.4f}")

#Generating signal
generator.eval()
with torch.no_grad():
    noise = generate_noise(1, latent_dim)
    generated_signal = generator(noise).detach().numpy().flatten()

#plots
plt.figure()
plt.plot(generated_signal)
plt.title("Generated EMG Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(data_p1_normalized[0:1000], label="Real Signal")
plt.plot(generated_signal, label="Generated Signal")
plt.title("Real and Generated Signals")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.title("Generator and Critic Loss")
plt.plot(G_losses, label="Generator Loss")
plt.plot(C_losses, label="Critic Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
