(.venv) D:\개발\1_ImageProcessingTool\ImageProcessingAPI\backend>python -m pip install -r requirements-openvino-diffusion.txt
Requirement already satisfied: optimum-intel<3,>=2.2 in .\.venv\Lib\site-packages (from optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (2.2.0)
Requirement already satisfied: openvino<2027,>=2026.3 in .\.venv\Lib\site-packages (from -r requirements-openvino-diffusion.txt (line 6)) (2026.4.0)
Collecting diffusers<0.50,>=0.40 (from -r requirements-openvino-diffusion.txt (line 7))
  Using cached diffusers-0.40.0-py3-none-any.whl.metadata (20 kB)
Requirement already satisfied: transformers<6,>=4.51 in .\.venv\Lib\site-packages (from -r requirements-openvino-diffusion.txt (line 8)) (5.5.4)
Collecting accelerate<2,>=1.5 (from -r requirements-openvino-diffusion.txt (line 9))
  Using cached accelerate-1.15.0-py3-none-any.whl.metadata (19 kB)
Requirement already satisfied: safetensors<1,>=0.5 in .\.venv\Lib\site-packages (from -r requirements-openvino-diffusion.txt (line 10)) (0.8.0)
Requirement already satisfied: torch>=2.1 in .\.venv\Lib\site-packages (from optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (2.14.0)
Requirement already satisfied: optimum~=2.3.0 in .\.venv\Lib\site-packages (from optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (2.3.0)
Requirement already satisfied: setuptools in .\.venv\Lib\site-packages (from optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (84.0.0)
Requirement already satisfied: huggingface-hub<1.22,>=0.23.2 in .\.venv\Lib\site-packages (from optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (1.21.0)
Requirement already satisfied: nncf>=3.3.0 in .\.venv\Lib\site-packages (from optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (3.4.0)
Requirement already satisfied: openvino-tokenizers>=2026.0 in .\.venv\Lib\site-packages (from optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (2026.4.0.0)
Requirement already satisfied: requests<3.0,>=2.33 in .\.venv\Lib\site-packages (from optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (2.34.2)
Requirement already satisfied: torchvision in .\.venv\Lib\site-packages (from optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (0.29.0)
Requirement already satisfied: numpy<2.6.0,>=1.16.6 in .\.venv\Lib\site-packages (from openvino<2027,>=2026.3->-r requirements-openvino-diffusion.txt (line 6)) (2.4.6)
Requirement already satisfied: openvino-telemetry>=2023.2.1 in .\.venv\Lib\site-packages (from openvino<2027,>=2026.3->-r requirements-openvino-diffusion.txt (line 6)) (2025.2.0)
Collecting importlib_metadata (from diffusers<0.50,>=0.40->-r requirements-openvino-diffusion.txt (line 7))
  Using cached importlib_metadata-9.0.1-py3-none-any.whl.metadata (4.5 kB)
Requirement already satisfied: filelock in .\.venv\Lib\site-packages (from diffusers<0.50,>=0.40->-r requirements-openvino-diffusion.txt (line 7)) (4.0.1)
Requirement already satisfied: httpx<1.0.0 in .\.venv\Lib\site-packages (from diffusers<0.50,>=0.40->-r requirements-openvino-diffusion.txt (line 7)) (0.28.1)
INFO: pip is looking at multiple versions of diffusers to determine which version is compatible with other requirements. This could take a while.
Collecting optimum-intel<3,>=2.2 (from optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5))
  Using cached optimum_intel-2.2.0-py3-none-any.whl.metadata (7.9 kB)
Collecting openvino<2027,>=2026.3 (from -r requirements-openvino-diffusion.txt (line 6))
  Using cached openvino-2026.4.0-22959-cp312-cp312-win_amd64.whl.metadata (13 kB)
Requirement already satisfied: click>=8.4.0 in .\.venv\Lib\site-packages (from huggingface-hub<1.22,>=0.23.2->optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (8.5.0)
Requirement already satisfied: fsspec>=2023.5.0 in .\.venv\Lib\site-packages (from huggingface-hub<1.22,>=0.23.2->optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (2026.9.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.5.1 in .\.venv\Lib\site-packages (from huggingface-hub<1.22,>=0.23.2->optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (1.6.0)
Requirement already satisfied: packaging>=20.9 in .\.venv\Lib\site-packages (from huggingface-hub<1.22,>=0.23.2->optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (26.3)
Requirement already satisfied: pyyaml>=5.1 in .\.venv\Lib\site-packages (from huggingface-hub<1.22,>=0.23.2->optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (6.0.3)
Requirement already satisfied: tqdm>=4.42.1 in .\.venv\Lib\site-packages (from huggingface-hub<1.22,>=0.23.2->optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (4.70.1)
Requirement already satisfied: typer<0.26.0,>=0.20.0 in .\.venv\Lib\site-packages (from huggingface-hub<1.22,>=0.23.2->optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (0.25.1)
Requirement already satisfied: typing-extensions>=4.1.0 in .\.venv\Lib\site-packages (from huggingface-hub<1.22,>=0.23.2->optimum-intel<3,>=2.2->optimum-intel[openvino]<3,>=2.2->-r requirements-openvino-diffusion.txt (line 5)) (4.16.0)
Collecting huggingface-hub<2.0,>=1.23.0 (from diffusers<0.50,>=0.40->-r requirements-openvino-diffusion.txt (line 7))
  Downloading huggingface_hub-1.32.0-py3-none-any.whl.metadata (16 kB)
Requirement already satisfied: regex!=2019.12.17 in .\.venv\Lib\site-packages (from diffusers<0.50,>=0.40->-r requirements-openvino-diffusion.txt (line 7)) (2026.9.10)
Requirement already satisfied: Pillow in .\.venv\Lib\site-packages (from diffusers<0.50,>=0.40->-r requirements-openvino-diffusion.txt (line 7)) (12.3.0)
INFO: pip is looking at multiple versions of optimum-intel to determine which version is compatible with other requirements. This could take a while.
ERROR: Cannot install -r requirements-openvino-diffusion.txt (line 7) and optimum-intel because these package versions have conflicting dependencies.

The conflict is caused by:
    diffusers 0.40.0 depends on huggingface-hub<2.0 and >=1.23.0
    optimum-intel 2.2.0 depends on huggingface-hub<1.22 and >=0.23.2

Additionally, some packages in these conflicts have no matching distributions available for your environment:
    huggingface-hub

To fix this you could try to:
1. loosen the range of package versions you've specified
2. remove package versions to allow pip to attempt to solve the dependency conflict

ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts
