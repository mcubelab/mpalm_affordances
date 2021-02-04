CUDA_VERSION=$1

echo "Cuda Version: ${CUDA_VERSION}"
pip install -r requirements.txt

if [[ ${CUDA_VERSION} == *"11"* ]]
then
    echo "Found CUDA Version: 11"
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html 
    pip install dgl-cu110
elif [[ ${CUDA_VERSION} == *"10"* ]] 
then
    echo "Found CUDA Version: 10"
    pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    pip install dgl-cu101
else
    echo "CUDA Version not recognized"
fi