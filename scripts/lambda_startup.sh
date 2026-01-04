sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev


python3.12 -m venv ~/py312
source ~/py312/bin/activate
python -m pip install --upgrade pip setuptools wheel

pip install -e .
