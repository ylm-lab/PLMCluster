
# First create a conda environment with python=3.12 installed.

conda create -n tmvec faiss-cpu python=3.12 -c pytorch
conda activate tmvec

# for GPU, if fails, install mkl via `conda install mkl=2021 mkl_fft`

conda create -n tmvec faiss-gpu python=3.12 -c pytorch

conda install pandas -c conda-forge
conda install pytorch torchvision torchaudio cpuonly -c pytorch

pip install pytorch_lightning
conda install transformers -c conda-forge
conda install pysam -c bioconda
pip install SentencePiece
