python prepare_FLMR.py

cd /home/jah649/mydata
mkdir EVQA
cd EVQA

wget https://vortexstorage7348269.blob.core.windows.net/flmrmodels/models_pipeline1.zip
wget https://vortexstorage7348269.blob.core.windows.net/flmrdata/EVQA_data.zip
wget https://vortexstorage7348269.blob.core.windows.net/flmrdata/EVQA_passages.zip
wget https://vortexstorage7348269.blob.core.windows.net/flmrdata/EVQA_test_split.tar.gz
wget https://vortexstorage7348269.blob.core.windows.net/flmrdata/EVQA_train_split.tar.gz
wget https://huggingface.co/datasets/BByrneLab/M2KR_Images/resolve/main/EVQA/google-landmark.tar
wget https://huggingface.co/datasets/BByrneLab/M2KR_Images/resolve/main/EVQA/inat.zip
echo "Finished wget"

mkdir google-landmark &&  tar -xvf google-landmark.tar -C google-landmark
mkdir inat &&  unzip inat.zip -d inat/
echo "Finished image root"

mkdir index
tar -xvzf EVQA_test_split.tar.gz
tar -xvzf EVQA_train_split.tar.gz
mv EVQA_test_split/ index/
mv EVQA_train_split/ index/
rm EVQA_test_split.tar.gz EVQA_train_split.tar.gz
echo "Finished index root"


mkdir EVQA_data &&  unzip EVQA_data.zip -d EVQA_data
mkdir EVQA_passages &&  unzip EVQA_passages.zip -d EVQA_passages
mkdir models &&  unzip models_pipeline1.zip -d models
mv models/models_pipeline1/*pt models/ &&  rm models_pipeline1.zip &&  rm -rf models/models_pipeline1/
echo "Finished Datasets and model ckpts"