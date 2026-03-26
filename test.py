import torch
import torch.nn as nn
import time

# --- CONFIGURATION ---
INPUT_FEATURES = 620
HIDDEN_LAYERS = 4
NEURONS_PER_LAYER = 512
BATCH_SIZE = 1024

class PINNSimulator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(INPUT_FEATURES, NEURONS_PER_LAYER), nn.Tanh()]
        for _ in range(HIDDEN_LAYERS):
            layers.extend([nn.Linear(NEURONS_PER_LAYER, NEURONS_PER_LAYER), nn.Tanh()])
        layers.append(nn.Linear(NEURONS_PER_LAYER, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def run_test():
    if not torch.cuda.is_available():
        print("CUDA not available. Check drivers.")
        return

    device = torch.device("cuda")
    print(f"Target GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM Allocated to Environment: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    model = PINNSimulator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(BATCH_SIZE, INPUT_FEATURES, device=device, requires_grad=True)

    print("\nInitiating Second-Order Derivative Stress Test...")
    try:
        start_time = time.time()
        for i in range(10):
            optimizer.zero_grad()
            u = model(x)
            
            # First derivative
            grad_u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
            # Second derivative (Physics Loss simulation - VRAM killer)
            grad_u_x = torch.autograd.grad(grad_u, x, torch.ones_like(grad_u), create_graph=True)[0]
            
            loss = torch.mean(grad_u_x**2)
            loss.backward()
            optimizer.step()
            
            vram_used = torch.cuda.memory_allocated(0) / 1e9
            print(f"Iteration {i+1}/10 | VRAM Active: {vram_used:.2f} GB")
            
        print(f"\nSUCCESS: Completed in {time.time() - start_time:.2f} seconds.")
    except RuntimeError as e:
        print(f"\nFAILURE (OOM): {e}")
        print("Conclusion: GPU partition lacks sufficient VRAM for 620-feature PINN equations.")

if __name__ == "__main__":
    run_test()