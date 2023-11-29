import torch

if __name__ == '__main__':
    torch.manual_seed(42)
    # noise = torch.randn(1024, 4, 64, 64)
    # torch.save(noise, 'noise_1024.pt')
    noise = torch.randn(1024, 4, 32, 32)
    torch.save(noise, 'noise_1024_32.pt')
