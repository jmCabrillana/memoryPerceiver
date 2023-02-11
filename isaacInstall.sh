git config --global user.email "jean_manuel.cabrillana@yahoo.fr"
git config --global user.name "jmCabrillana"
/isaac-sim/python.sh -m pip install torch torchvision torchaudio
/isaac-sim/python.sh -m pip install -r requirements.txt
apt-get install curl tar
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-417.0.1-linux-x86_64.tar.gz
tar -xf google-cloud-cli-417.0.1-linux-x86_64.tar.gz 
./google-cloud-sdk/install.sh
/isaac-sim/python.sh -m pip install jupyter
#git token: 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