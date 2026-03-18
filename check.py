"""
GPU環境診断スクリプト
使い方: python check_gpu_env.py
必要なパッケージ: なし（標準ライブラリのみ）
"""
import subprocess
import sys

def run_command(cmd):
    """コマンドを実行して結果を返す"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None

def check_nvidia_driver():
    """NVIDIAドライバのバージョンを確認"""
    output = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    if output:
        print(f"[OK] NVIDIA Driver: {output}")
        return output
    else:
        print("[NG] NVIDIA Driver: 検出できません")
        return None

def check_cuda_version():
    """nvidia-smiから対応CUDA バージョンを確認"""
    output = run_command("nvidia-smi --query-gpu=compute_cap --format=csv,noheader")
    cc = output if output else "不明"
    
    # nvidia-smiのヘッダーからCUDAバージョンを取得
    smi_output = run_command("nvidia-smi")
    if smi_output and "CUDA Version:" in smi_output:
        for line in smi_output.split("\n"):
            if "CUDA Version:" in line:
                cuda_ver = line.split("CUDA Version:")[1].strip().split()[0]
                print(f"[OK] 対応CUDA (ドライバ上限): {cuda_ver}")
                print(f"[OK] Compute Capability: {cc}")
                return cuda_ver
    print(f"[--] Compute Capability: {cc}")
    return None

def check_pytorch():
    """PyTorchのインストール状況を確認"""
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
        print(f"[OK] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA version (PyTorch): {torch.version.cuda}")
            print(f"[OK] cuDNN version: {torch.backends.cudnn.version()}")
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("[--] PyTorch: インストールされていません")
        return False

def main():
    print("=" * 50)
    print("GPU環境診断レポート")
    print("=" * 50)
    print(f"\nPython: {sys.version}")
    print()
    
    check_nvidia_driver()
    check_cuda_version()
    print()
    check_pytorch()
    
    print("\n" + "=" * 50)
    print("診断完了")
    print("=" * 50)

if __name__ == "__main__":
    main()
