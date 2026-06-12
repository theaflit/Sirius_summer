import torch

def test_cuda():
    if torch.cuda.is_available():
        print("Доступный GPU:", torch.cuda.get_device_name(0))
        print("CUDA версия:", torch.version.cuda)

        return True
    
    return False

if __name__ == "__main__":
    test_cuda()