#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# git
git config --global user.email "jean_manuel.cabrillana@yahoo.fr"
git config --global user.name "jmCabrillana"

# directories and system
mkdir $SCRIPT_DIR/memory_perceiver/main/weights
mkdir $SCRIPT_DIR/memory_perceiver/main/monitor
mkdir $SCRIPT_DIR/memory_perceiver/main/experiments
sudo apt install cmake build-essential -y
apt-get install curl tar screen -y

# GCP
# curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-417.0.1-linux-x86_64.tar.gz
# tar -xf google-cloud-cli-417.0.1-linux-x86_64.tar.gz 
# rm google-cloud-cli-417.0.1-linux-x86_64.tar.gz 
# CLOUDSDK_CORE_DISABLE_PROMPTS=1 ./google-cloud-sdk/install.sh
# ./google-cloud-sdk/bin/gcloud --quiet auth activate-service-account --key-file=memoryPerceiver/service_account.json

# python and requirements
# alias python='/isaac-sim/python.sh'
if [ "$SCRIPT_DIR" = "/isaac-sim/memoryPerceiver" ];
then
    python_exe="/isaac-sim/kit/python/bin/python3"
    $python_exe -m pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    python3.10 -m venv isaaclab
    source $SCRIPT_DIR/../isaaclab/bin/activate
    cuda_version="$(nvcc --version | grep -oP '\d+\.\d+' | tr -d '.' | uniq)"
    echo $LD_LIBRARY_PATH
    echo $cuda_version
    python_exe="python"
    $python_exe -m pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu$cuda_version
    export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
    $python_exe -m pip install isaacsim[all]==4.5.0 --extra-index-url https://pypi.nvidia.com
    $python_exe -m pip install isaacsim[extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com
fi
$python_exe -m pip install --upgrade pip
$python_exe -m pip install -r $SCRIPT_DIR/requirements.txt
cd $SCRIPT_DIR/IsaacLab/source/isaaclab
$python_exe -m pip install -e .
cd $SCRIPT_DIR/IsaacLab/source/isaaclab_assets
$python_exe -m pip install -e .
cd $SCRIPT_DIR/IsaacLab/source/isaaclab_rl
$python_exe -m pip install -e .
cd $SCRIPT_DIR/IsaacLab/source/isaaclab_tasks
$python_exe -m pip install -e .

# Infinigen
sudo apt-get install wget cmake g++ libgles2-mesa-dev libglew-dev libglfw3-dev libglm-dev zlib1g-dev -y
sudo apt install libxkbcommon-x11-0 -y
sudo apt install python3.11 -y
sudo apt install python3.11-dev -y
sudo apt install python3.11-venv -y
cd $SCRIPT_DIR/..
python3.11 -m venv infinigen
source $SCRIPT_DIR/../infinigen/bin/activate
cd $SCRIPT_DIR/infinigen
python -m pip install -e ".[terrain,vis]"

# Set variables and update bashrc
export ACCEPT_EULA=Y
export ISAAC_JUPYTER_KERNEL=1
echo "source isaaclab/bin/activate" >> /home/user/.bashrc
echo "export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH" >> /home/user/.bashrc

# if nvidia icd config files are duplicated and not empty, truncate the first one (see memo.txt)
if [[ -s /etc/vulkan/icd.d/nvidia_icd.json && -s /usr/share/vulkan/icd.d/nvidia_icd.json ]]
    then > /etc/vulkan/icd.d/nvidia_icd.json
fi

#git token: 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
