import torch

def check_device(device: str):
    device = device.lower()

    if device not in ('gpu', 'cuda', 'cpu'):
        raise ValueError("device должен быть GPU или CPU")

    if device == 'cpu':
        return 'cpu'

    return 'cuda' if test_cuda() else 'cpu'

def test_cuda():
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        
        print("Доступный GPU:", green_text(device))
        print("CUDA версия:", torch.version.cuda)

        return True
    
    print("GPU не найден, поэтому будет использова CPU")
    return False

def green_text(text):
    return '\033[92m' + text + '\033[0m'

if __name__ == "__main__":
    test_cuda()