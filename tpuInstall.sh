git config --global user.email "jean_manuel.cabrillana@yahoo.fr"
git config --global user.name "jmCabrillana"
sudo apt update
sudo apt install ffmpeg -y
sudo apt-get install psmisc -y
pip install --upgrade pip
pip install torchvision==0.12.0 torchaudio==0.11.0 jupyter
cd hand4whole
chmod +x install.sh
./install.sh
cd ..
pip install -r requirements.txt
jupyter nbextension enable --py widgetsnbextension
#gcloud storage cp --recursive  gs://karagoz_persistence_1/data_test .
# gcloud storage cp gs://karagoz_persistence_1/Test_names.txt memory_perceiver/data_test/hist
# git clone https://github.com/facebookresearch/pytorch3d.git
# cd pytorch3d && pip install -e .