bash
git clone https://github.com/ShivamShrirao/diffusers.git
cd diffusers/examples/dreambooth 
mkdir ./reference
mkdir ./training_images
mkdir ./output
git config --global credential.helper store
#My bucket with the training images
aws s3 cp s3://maxencep-ml/teddy/ ./training_images --recursive
wget https://raw.githubusercontent.com/Victarry/stable-dreambooth/main/environment.yaml
conda env create -f environment.yaml
conda activate stable-diffusion
pip install git+https://github.com/ShivamShrirao/diffusers.git
pip install -U -r requirements.txt
#pip install git+https://github.com/facebookresearch/xformers@1d31a3a
#pip install xformers
pip install triton
pip install bitsandbytes
accelerate config
rm train_dreambooth.py
wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth.py