## Install (WSL / Ubuntu)

sudo apt update
sudo apt install -y python3 python3-pip python3-venv git

mkdir -p ~/apps/textfilter
cd ~/apps/textfilter
python3 -m venv .venv
source .venv/bin/activate

pip install git+https://github.com/zipperson/textfilter.git

echo "badword" > terms.txt
textfilter --terms terms.txt