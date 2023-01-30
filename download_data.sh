apt-get update && apt-get install -y wget git p7zip-full rsync

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
7z x data.zip -y -o.tmp
7z x Cascade_RCNN_model.zip -y -o.tmp
7z x Faster_RCNN_model.zip -y -o.tmp
echo "Moving data to folders..."
# Create data and models folder if necessary
# Copy all downloaded files into those folders
mkdir -p tutorial_exps
mkdir -p data
echo "Moving models to folders..."
rsync -ah --progress -r .tmp/data/ data/
rsync -ah --progress -r tmp/Cascade_RCNN_model/ tutorial_exps/Cascade_RCNN_model/
rsync -ah --progress -r tmp/Faster_RCNN_data_augmentation/ tutorial_exps/Faster_RCNN_data_augmentation/

cd data/default_annotations
mv -f train.csv data_train.csv && mv -f valid.csv data_val.csv && mv -f test.csv data_test.csv
cd ../../

# remove all temporary files
echo "Cleaning up..."
rm -rf .tmp
rm data.zip cascade_rcnn.zip faster_rcnn.zip
echo "Finished"