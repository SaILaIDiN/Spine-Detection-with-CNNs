import os
import shutil
import zipfile

import gdown


def prepare_data():
    fileids = ["1yi2tQ-0oJhElaSUDFn_UpZ-bUO0bH3_N", "1eLeqafL4UPM3-uPSV37SvEo00ONM-1tD", "1OnbBMdaOsc9-TPFkOTr4geCNLT5Dv7w3"]
    filenames = ["data.zip", "cascade_rcnn.zip", "faster_rcnn.zip"]
    foldernames = ["data", "Cascade_RCNN_model", "Faster_RCNN_data_augmentation"]
    new_basefolders = [".", "tutorial_exps", "tutorial_exps"]

    os.makedirs("tutorial_exps", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    for (fileid, filename, foldername, new_basefolder) in zip(fileids, filenames, foldernames, new_basefolders):
        if os.path.exists(filename):
            print(f"File {filename} already exists. Not downloaded again.")
            continue
        else:
            print(f"Start downloading {filename}")
            url = f"https://drive.google.com/uc?id={fileid}" #&confirm=""
            gdown.download(url, filename, resume=True)
        
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(".tmp")

        print(f"Moving {foldername} to folders ...")
        all_files = os.listdir(f".tmp/{foldername}")
        for path, subdirs, files in os.walk(f".tmp/{foldername}"):
            # split path into all subdirectories and remove the first two
            sub_folders = os.path.normpath(path).split(os.sep)[2:]
            if len(sub_folders) > 0:
                sub_path = os.path.join(*sub_folders)
            else:
                sub_path = ""
            target_folder = os.path.join(new_basefolder, foldername, sub_path)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder, exist_ok=True)
            for name in files:
                shutil.move(os.path.join(path, name), os.path.join(target_folder, name))

        if foldername == "data":
            # rename train, val and test files
            rename_files = ["train.csv", "valid.csv", "test.csv"]
            new_filenames = ["data_train.csv", "data_val.csv", "data_test.csv"]
            base_folder = "data/default_annotations"
            for (rename_file, new_filename) in zip(rename_files, new_filenames):
                shutil.move(os.path.join(base_folder, rename_file), os.path.join(base_folder, new_filename))
        print("Finished with moving.")
    print("Cleaning up")
    if os.path.exists(".tmp"):
        shutil.rmtree(".tmp")

if __name__ == "__main__":
    prepare_data()