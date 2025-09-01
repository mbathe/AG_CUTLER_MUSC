#!/usr/bin/env python3

import torch
import subprocess

def check_gpu_setup():
    """Vérifie la configuration GPU complète"""
    print("=== VÉRIFICATION GPU POUR CUTLER ===\n")
    
    # 1. CUDA disponible dans PyTorch
    print("1. PyTorch CUDA:")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA disponible: {cuda_available}")
    
    if cuda_available:
        print(f"   Version CUDA: {torch.version.cuda}")
        print(f"   Nombre de GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name}")
            print(f"   VRAM: {props.total_memory / 1e9:.1f} GB")
    else:
        print("   ❌ CUDA non disponible dans PyTorch")
    
    # 2. NVIDIA driver
    print("\n2. NVIDIA Driver:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"   ✅ {line.strip()}")
                    break
        else:
            print("   ❌ nvidia-smi non trouvé")
    except FileNotFoundError:
        print("   ❌ NVIDIA driver non installé")
    
    # 3. Test simple GPU
    print("\n3. Test GPU:")
    if cuda_available:
        try:
            x = torch.randn(100, 100).cuda()
            y = x @ x.T
            print("   ✅ Test tensor GPU réussi")
        except Exception as e:
            print(f"   ❌ Erreur test GPU: {e}")
    
    # 4. Recommandations
    print("\n4. Recommandations:")
    if not cuda_available:
        print("   Pour activer le GPU:")
        print("   1. Installer NVIDIA drivers")
        print("   2. Installer CUDA toolkit")
        print("   3. Réinstaller PyTorch avec CUDA:")
        print("      conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    else:
        print("   ✅ Configuration GPU prête!")
        print("   Pour utiliser avec CutLER:")
        print("   python tools/train_defect_detection.py --dataset-path ./dataset_defect --output-dir ./output_gpu")

if __name__ == "__main__":
    check_gpu_setup()
