# git
git config --global user.email "jean_manuel.cabrillana@yahoo.fr"
git config --global user.name "jmCabrillana"

# directories and system
mkdir memory_perceiver/main/weights
mkdir memory_perceiver/main/monitor
mkdir memory_perceiver/main/experiments
sudo apt install cmake build-essential -y
apt-get install curl tar screen -y

# GCP
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-417.0.1-linux-x86_64.tar.gz
tar -xf google-cloud-cli-417.0.1-linux-x86_64.tar.gz 
rm google-cloud-cli-417.0.1-linux-x86_64.tar.gz 
CLOUDSDK_CORE_DISABLE_PROMPTS=1 ./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud --quiet auth activate-service-account --key-file=/isaac-sim/memoryPerceiver/service_account.json

# python and requirements
# alias python='/isaac-sim/python.sh'
/isaac-sim/python.sh -m pip install torch torchvision torchaudio
/isaac-sim/python.sh -m pip install -r requirements.txt
cd /isaac-sim/memoryPerceiver/IsaacLab/source/extensions/omni.isaac.lab
/isaac-sim/python.sh -m pip install -e .
cd /isaac-sim/memoryPerceiver/IsaacLab/source/extensions/omni.isaac.lab_assets
/isaac-sim/python.sh -m pip install -e .
cd /isaac-sim/memoryPerceiver/IsaacLab/source/extensions/omni.isaac.lab_tasks
/isaac-sim/python.sh -m pip install -e .

# Infinigen
sudo apt-get install wget cmake g++ libgles2-mesa-dev libglew-dev libglfw3-dev libglm-dev zlib1g-dev -y
sudo apt install libxkbcommon-x11-0
cd /isaac-sim/memoryPerceiver/infinigen
/isaac-sim/python.sh -m pip install -e ".[terrain,vis]"

# Set variables
export ACCEPT_EULA=Y
export ISAAC_JUPYTER_KERNEL=1

# sed -i '128,135s/^/#/' /isaac-sim/kit/exts/omni.kit.widget.toolbar/omni/kit/widget/toolbar/builtin_tools/play_button_group.py
# sed -i '127a\        pass' /isaac-sim/kit/exts/omni.kit.widget.toolbar/omni/kit/widget/toolbar/builtin_tools/play_button_group.py
# cat /isaac-sim/memoryPerceiver/memory_perceiver/isaac/camera_missing.txt >> /isaac-sim/exts/omni.isaac.core/omni/isaac/core/utils/torch/rotations.py

#  MEMO
#./google-cloud-sdk/bin/gcloud --quiet config set project turnkey-setup-356207
#  ./google-cloud-sdk/bin/gcloud storage cp --recursive  gs://memoryPerceiver/experiments .
#git token: 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