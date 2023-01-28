echo "Starting downloading data..."
# Download data used for training, validation and testing as well as the record files
fileid=1yi2tQ-0oJhElaSUDFn_UpZ-bUO0bH3_N; filename=data.zip;

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id="$fileid"&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$fileid' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')" -O $filename && rm -rf /tmp/cookies.txt
echo "Starting downloading model..."
# Download model.ckpt and frozen_inference_graph.pb files (without saved_model.pb)
# ATTENTION: First subfolder is called own_models2!
fileid=1IqXEYAbruormi9g354a1MtugQJQiZKGL; filename=model.zip;

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id="$fileid"&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$fileid' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')" -O $filename && rm -rf /tmp/cookies.txt
echo "Finished download. Unzipping..."
unzip data.zip -d .tmp
unzip model.zip -d .tmp
echo "Moving data to folders..."
# Create data and models folder if necessary
# Copy all downloaded files into those folders
mkdir -p data/raw/
mv -f .tmp/data/raw/* data/raw/

mkdir -p data/default_annotations/
mv -f .tmp/data/default_annotations/* data/default_annotations/
cd data/default_annotations
mv -f train.csv data_train.csv && mv -f valid.csv data_val.csv && mv -f test.csv data_test.csv
cd ../../

mkdir -p own_models/default_model/
mv -f .tmp/own_models2/default_model/* own_models/default_model/

# remove all temporary files
echo "Cleaning up..."
rm -rf .tmp
rm data.zip
rm model.zip
echo "Finished"