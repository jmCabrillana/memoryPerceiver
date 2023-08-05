git config --global user.email "jean_manuel.cabrillana@yahoo.fr"
git config --global user.name "jmCabrillana"
mkdir memory_perceiver/main/weights
mkdir memory_perceiver/main/outputs
mkdir memory_perceiver/main/monitor
/isaac-sim/python.sh -m pip install torch torchvision torchaudio
/isaac-sim/python.sh -m pip install -r requirements.txt
apt-get install curl tar
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-417.0.1-linux-x86_64.tar.gz
tar -xf google-cloud-cli-417.0.1-linux-x86_64.tar.gz 
./google-cloud-sdk/install.sh
cd OmniIsaacGymEnvs
/isaac-sim/python.sh -m pip install -e .
#git token: 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