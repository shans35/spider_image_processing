# creates environment, installs pytorch and Segment Anything Model ver. 2 from Github
# (cuda version is specific to lab linux)

conda env create -f environment.yml
conda activate env_swag

export UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=1
uv pip install torch torchvision --index https://download.pytorch.org/whl/cu126 --reinstall
uv pip install 'git+https://github.com/facebookresearch/sam2.git'