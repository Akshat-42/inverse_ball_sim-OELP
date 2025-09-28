import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

import FrEIA.framework as Ff
import FrEIA.modules as Fm

# --- 1. Configuration & Setup ---
# Define the dimensions based on our vector-based approach
# Bowling Params (x): [vx, vy, vz, wx, wy, wz]
# Outcome Space (z): [land_x, land_y, lat1, lat2, lat3, lat4]
TOTAL_DIMS = 6
LANDING_SPOT_DIMS = 2
LATENT_DIMS = TOTAL_DIMS - LANDING_SPOT_DIMS

# Model & Training Hyperparameters
HIDDEN_DIM = 256
NUM_LAYERS = 8
EPOCHS = 2000
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_SAMPLES_TRAIN = 8192 # More data is better

# --- 2. Dummy Physics Simulator ---
# IMPORTANT: Replace this with your actual, accurate physics engine.
# This simple polynomial function is just a placeholder to make the code runnable.
def physics_simulator(bowling_params_x):
    """
    Takes a batch of bowling parameters [N, 6] and returns landing spots [N, 2].
    This is a placeholder. Your real physics engine will be much more complex.
    """
    vx = bowling_params_x[:, 0]
    vz = bowling_params_x[:, 2]
    wy = bowling_params_x[:, 4]
    
    # A simple, non-linear relationship for demonstration
    land_x = 0.1 * vx - 0.05 * vz + 2.0  # Velocity and angle affect length
    land_y = 0.0001 * wy + 0.01 * vx - 1.4 # Spin and velocity affect side movement
    
    return torch.stack((land_x, land_y), dim=1)

def generate_training_data(num_samples):
    """Generates the training dataset by calling the simulator."""
    # Generate random but plausible bowling parameters
    # [vx, vy, vz, wx, wy, wz]
    bowling_params_x = torch.cat([
        torch.randn(num_samples, 1) * 10 + 130, # vx (speed)
        torch.randn(num_samples, 1) * 5 - 2.5,   # vy (sideways velocity)
        torch.randn(num_samples, 1) * 4 + 8,     # vz (vertical velocity)
        torch.randn(num_samples, 1) * 3000,      # wx (spin)
        torch.randn(num_samples, 1) * 3000,      # wy (spin)
        torch.randn(num_samples, 1) * 3000       # wz (spin)
    ], dim=1)
    
    # Get the "ground truth" landing spots from the physics engine
    landing_spots_z_cond = physics_simulator(bowling_params_x)
    
    return bowling_params_x, landing_spots_z_cond

# --- 3. Model Definition ---
def define_inn_model():
    """Defines the Invertible Neural Network architecture using FrEIA."""
    def subnet_fc(c_in, c_out):
        return nn.Sequential(
            nn.Linear(c_in, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, c_out)
        )

    model = Ff.SequenceINN(TOTAL_DIMS)
    for i in range(NUM_LAYERS):
        model.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    
    return model

# --- 4. Training Loop ---
def train(model, train_loader, device):
    """Trains the model using the forward pass."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    print("--- Starting Training ---")
    
    for epoch in range(EPOCHS):
        print("epoch: ",epoch)
        total_loss = 0
        for x_batch, z_cond_batch in train_loader:
            x_batch, z_cond_batch = x_batch.to(device), z_cond_batch.to(device)
            optimizer.zero_grad()
            
            # FORWARD PASS: Map bowling params 'x' to outcome space 'z'
            z_pred, log_det_j = model(x_batch)
            
            # Custom Loss Function:
            # 1. We want the first 2 dims of z_pred to match the ground truth landing spot.
            loss_fit = torch.mean((z_pred[:, :LANDING_SPOT_DIMS] - z_cond_batch)**2)
            # 2. We want the latent variables to be a simple Gaussian distribution.
            loss_latent = torch.mean(z_pred[:, LANDING_SPOT_DIMS:]**2)
            
            # Combine the losses
            loss = loss_fit + 0.1 * loss_latent # Weighting can be tuned
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")
    print("--- Training Complete ---")

# --- 5. Prediction & Verification ---
def verify_model(model, device):
    """Demonstrates using the model in reverse and verifies the output."""
    model.eval()
    print("\n--- Verifying Model (Inverse Pass) ---")
    
    # Define our strategic targets (from the Verification Dataset)
    verification_targets = {
        "Good Length, Off Stump": torch.tensor([[6.5, 0.15]]),
        "Yorker, Leg Stump": torch.tensor([[1.9, -0.10]])
    }

    for name, target_spot in verification_targets.items():
        print(f"\nTargeting: '{name}' at {target_spot.numpy()}")
        
        # Combine target spot with random latent variables to form the input 'z'
        random_style = torch.randn(1, LATENT_DIMS, device=device)
        z_input = torch.cat((target_spot.to(device), random_style), dim=1)
        
        # INVERSE PASS: Get bowling instructions
        predicted_params, _ = model(z_input, rev=True)
        
        print("  Predicted Bowling Params [vx,vy,vz, wx,wy,wz]:")
        print(f"  {np.round(predicted_params.cpu().detach().numpy(), 2)}")
        
        # FINAL CHECK: Use the physics simulator to see where the predicted params land
        achieved_spot = physics_simulator(predicted_params.cpu().detach())
        print(f"  Result from Physics Sim: {np.round(achieved_spot.numpy(), 2)}")

        error = torch.sum((target_spot - achieved_spot)**2).item()
        print(f"  Landing Spot Error: {error:.4f}")

# --- Main Execution ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # 1. Generate Data
    x_train, z_cond_train = generate_training_data(NUM_SAMPLES_TRAIN)
    train_dataset = TensorDataset(x_train, z_cond_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Define and Train Model
    inn_model = define_inn_model().to(device)
    # train(inn_model, train_loader, device)

    # 3. Verify Predictions
    verify_model(inn_model, device)
