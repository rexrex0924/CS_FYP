"""
Setup script to help install and configure Ollama with models for positional bias testing
"""

import os
import platform
import subprocess
import sys
import time
import requests
from typing import List


def check_ollama_installed() -> bool:
    """Check if Ollama is installed and accessible"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_ollama_running() -> bool:
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_available_models() -> List[str]:
    """List models available in Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except requests.exceptions.RequestException:
        pass
    return []


def pull_model(model_name: str) -> bool:
    """Pull a model using Ollama"""
    print(f"Pulling model: {model_name}")
    print("This may take several minutes depending on model size...")
    
    try:
        # Use ollama pull command
        result = subprocess.run(
            ["ollama", "pull", model_name], 
            capture_output=False,  # Show output in real-time
            text=True,
            timeout=1800  # 30 minute timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Timeout while pulling {model_name}")
        return False
    except Exception as e:
        print(f"Error pulling {model_name}: {e}")
        return False


def recommend_models():
    """Recommend models for positional bias testing"""
    recommendations = [
        {
            "name": "llama3.2:3b",
            "description": "Small, fast Llama 3.2 model (3B parameters, ~2GB)",
            "size": "~2GB"
        },
        {
            "name": "mistral:7b",
            "description": "Mistral 7B model - good balance of performance and size",
            "size": "~4GB"
        },
        {
            "name": "qwen2.5:7b",
            "description": "Qwen 2.5 7B model - alternative architecture",
            "size": "~4GB"
        },
        {
            "name": "phi3:mini",
            "description": "Microsoft Phi-3 Mini - very small and fast",
            "size": "~2GB"
        }
    ]
    
    print("\n🤖 RECOMMENDED MODELS FOR POSITIONAL BIAS TESTING:")
    print("=" * 60)
    for i, model in enumerate(recommendations, 1):
        print(f"{i}. {model['name']}")
        print(f"   Description: {model['description']}")
        print(f"   Size: {model['size']}")
        print()


def main():
    print("🔧 OLLAMA SETUP FOR POSITIONAL BIAS TESTING")
    print("=" * 50)
    
    # Check if Windows
    if platform.system() != "Windows":
        print("⚠️  This script is designed for Windows. For other OS, visit: https://ollama.com")
        return
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("❌ Ollama not found!")
        print("\n📥 INSTALLATION STEPS:")
        print("1. Go to https://ollama.com/download/windows")
        print("2. Download and run the Windows installer")
        print("3. Restart this script after installation")
        print("\nOr manually install via PowerShell:")
        print("   winget install Ollama.Ollama")
        return
    
    print("✅ Ollama is installed!")
    
    # Check if Ollama is running
    if not check_ollama_running():
        print("⚠️  Ollama service is not running")
        print("\n🚀 STARTING OLLAMA:")
        print("Trying to start Ollama service...")
        
        try:
            # Try to start Ollama
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
            print("Waiting for Ollama to start...")
            time.sleep(5)
            
            if check_ollama_running():
                print("✅ Ollama is now running!")
            else:
                print("❌ Failed to start Ollama automatically")
                print("Please start Ollama manually:")
                print("1. Open Command Prompt or PowerShell")
                print("2. Run: ollama serve")
                print("3. Keep that window open and re-run this script")
                return
        except Exception as e:
            print(f"Error starting Ollama: {e}")
            return
    else:
        print("✅ Ollama service is running!")
    
    # List current models
    current_models = list_available_models()
    if current_models:
        print(f"\n📚 CURRENTLY INSTALLED MODELS:")
        for model in current_models:
            print(f"   • {model}")
    else:
        print("\n📚 No models currently installed")
    
    # Show recommendations
    recommend_models()
    
    # Interactive model installation
    print("🔽 INSTALL MODELS:")
    print("Enter model names to install (one per line, or 'q' to quit):")
    print("Examples: llama3.2:3b, mistral:7b, phi3:mini")
    
    while True:
        try:
            model_input = input("\nModel to install (or 'q' to quit): ").strip()
            
            if model_input.lower() in ['q', 'quit', 'exit']:
                break
            
            if not model_input:
                continue
            
            if model_input in current_models:
                print(f"✅ {model_input} is already installed!")
                continue
            
            success = pull_model(model_input)
            if success:
                print(f"✅ Successfully installed {model_input}")
                current_models.append(model_input)
            else:
                print(f"❌ Failed to install {model_input}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Final status
    final_models = list_available_models()
    if final_models:
        print(f"\n🎉 SETUP COMPLETE!")
        print(f"Installed models: {', '.join(final_models)}")
        print(f"\nYou can now run positional bias evaluation:")
        print(f"python eval_positional_bias.py --model {final_models[0]} --input data/sample_mcq.csv")
    else:
        print("\n⚠️  No models installed. Please install at least one model to continue.")


if __name__ == "__main__":
    main()
