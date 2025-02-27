python prepare_FLMR.py

cd /mydata
sudo mkdir EVQA
cd EVQA

sudo wget https://vortexstorage7348269.blob.core.windows.net/flmrmodels/models_pipeline1.zip
sudo wget https://vortexstorage7348269.blob.core.windows.net/flmrdata/EVQA_data.zip
sudo wget https://vortexstorage7348269.blob.core.windows.net/flmrdata/EVQA_passages.zip
sudo wget https://vortexstorage7348269.blob.core.windows.net/flmrdata/EVQA_test_split.tar.gz
sudo wget https://vortexstorage7348269.blob.core.windows.net/flmrdata/EVQA_train_split.tar.gz
sudo wget https://huggingface.co/datasets/BByrneLab/M2KR_Images/resolve/main/EVQA/google-landmark.tar
sudo wget https://huggingface.co/datasets/BByrneLab/M2KR_Images/resolve/main/EVQA/inat.zip
echo "Finished wget"

sudo mkdir google-landmark && sudo tar -xvf google-landmark.tar -C google-landmark
sudo mkdir inat && sudo unzip inat.zip -d inat/
echo "Finished image root"

sudo mkdir index
sudo tar -xvzf EVQA_test_split.tar.gz
sudo tar -xvzf EVQA_train_split.tar.gz
sudo mv EVQA_test_split/ index/
sudo mv EVQA_train_split/ index/
sudo rm EVQA_test_split.tar.gz EVQA_train_split.tar.gz
echo "Finished index root"


sudo mkdir EVQA_data && sudo unzip EVQA_data.zip -d EVQA_data
sudo mkdir EVQA_passages && sudo unzip EVQA_passages.zip -d EVQA_passages
sudo mkdir models && sudo unzip models_pipeline1.zip -d models
sudo mv models/models_pipeline1/*pt models/ && sudo rm models_pipeline1.zip && sudo rm -rf models/models_pipeline1/
echo "Finished Datasets and model ckpts"