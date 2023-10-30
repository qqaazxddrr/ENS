@echo off
call conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
call conda install -c anaconda pandas
call conda install -c anaconda scikit-learn
call conda install -c conda-forge tqdm
call conda install -c anaconda numpy
call conda install -c conda-forge matplotlib



