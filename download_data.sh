echo "Starting downloading data..."
# Download data used for training, validation and testing as well as the record files
wget "https://drive.google.com/uc?export=download&id=1yi2tQ-0oJhElaSUDFn_UpZ-bUO0bH3_N" -O data.zip # -o data.zip
echo "Starting downloading model..."
# Download model.ckpt and frozen_inference_graph.pb files (without saved_model.pb)
# ATTENTION: First subfolder is called own_models2!
wget "https://drive.google.com/uc?export=download&id=1IqXEYAbruormi9g354a1MtugQJQiZKGL" -O model.zip
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

mkdir -p own_models/default_model/
mv -f .tmp/own_models2/default_model/* own_models/default_model/

# remove all temporary files
echo "Cleaning up..."
rm -rf .tmp
rm data.zip
echo "Finished"