git config --global user.email "jean_manuel.cabrillana@yahoo.fr"
git config --global user.name "jmCabrillana"
sudo apt update
sudo apt install ffmpeg -y
sudo apt-get install psmisc -y
conda create python=3.8 -n karagoz -y
source ~/.bashrc
conda activate karagoz
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
conda install pip -y
cd hand4whole
chmod +x install.sh
./install.sh
cd ..
pip install -r requirements.txt
pip install jupyter
jupyter nbextension enable --py widgetsnbextension