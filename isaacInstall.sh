git config --global user.email "jean_manuel.cabrillana@yahoo.fr"
git config --global user.name "jmCabrillana"
mkdir memory_perceiver/main/weights
mkdir memory_perceiver/main/outputs
mkdir memory_perceiver/main/monitor
/isaac-sim/python.sh -m pip install torch torchvision torchaudio
/isaac-sim/python.sh -m pip install -r requirements.txt
apt-get install curl tar -y
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-417.0.1-linux-x86_64.tar.gz
tar -xf google-cloud-cli-417.0.1-linux-x86_64.tar.gz 
CLOUDSDK_CORE_DISABLE_PROMPTS=1 ./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud --quiet auth activate-service-account --key-file=/isaac-sim/memoryPerceiver/turnkey-setup-356207-eaf9509e1a1f.json
rm google-cloud-cli-417.0.1-linux-x86_64.tar.gz 
cd OmniIsaacGymEnvs
/isaac-sim/python.sh -m pip install -e .
export ACCEPT_EULA=Y
export ISAAC_JUPYTER_KERNEL=1
#./google-cloud-sdk/bin/gcloud --quiet config set project turnkey-setup-356207
#  ./google-cloud-sdk/bin/gcloud storage cp --recursive  gs://karagoz_persistence_1/data_test .
#git token: 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