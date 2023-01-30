apt-get update && apt-get install wget git
echo "Starting downloading data..."
# Download data used for training, validation and testing as well as the record files
fileid=1yi2tQ-0oJhElaSUDFn_UpZ-bUO0bH3_N; filename=data.zip;

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id="$fileid"&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$fileid' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')" -O $filename && rm -rf /tmp/cookies.txt
echo "Starting downloading models..."

# Download Cascade RCNN
fileid=1eLeqafL4UPM3-uPSV37SvEo00ONM-1tD; filename=cascade_rcnn.zip;
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id="$fileid"&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$fileid' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')" -O $filename && rm -rf /tmp/cookies.txt

# Download Faster RCNN
fileid=1OnbBMdaOsc9-TPFkOTr4geCNLT5Dv7w3; filename=faster_rcnn.zip;
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id="$fileid"&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$fileid' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')" -O $filename && rm -rf /tmp/cookies.txt

echo "Finished download. Unzipping..."
unzip data.zip -d .tmp
unzip cascade_rcnn.zip -d .tmp
unzip faster_rcnn.zip -d .tmp
echo "Moving data to folders..."
# Create data and models folder if necessary
# Copy all downloaded files into those folders
mv -f .tmp/data/ data/
mkdir -p tutorial_exps
mv -f .tmp/Cascade_RCNN_model tutorial_exps/Cascade_RCNN_model
mv -f .tmp/Faster_RCNN_data_augmentation tutorial_exps/Faster_RCNN_data_augmentation

cd data/default_annotations
mv -f train.csv data_train.csv && mv -f valid.csv data_val.csv && mv -f test.csv data_test.csv
cd ../../

# mkdir -p own_models/default_model/
# mv -f .tmp/own_models2/default_model/* own_models/default_model/

# remove all temporary files
echo "Cleaning up..."
rm -rf .tmp
rm data.zip
# rm model.zip
echo "Finished"