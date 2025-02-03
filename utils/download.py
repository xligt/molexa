import os, gdown, zipfile, pickle

def download_file_from_google_drive(file_id, destination):
    
    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        gdown.download(url, destination, quiet=False)
        print(f"File downloaded successfully to {destination}")
    except Exception as e:
        print(f"An error occurred while downloading the file: {e}")


def download_dataset_weights():
    download_folder = '../download/'
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    id_dict = {'dataset': '1TR_bS2GRgz-HqnP_NA566y1eogbORw8F', 'model': '114-Qe5rrJc4nghPRBZ9_xEqK1XteJ2bb'}
    for k in id_dict.keys():
        dest =  download_folder + k
        zip_file_path = dest +'.zip'
        extract_to = dest if k=='model' else download_folder
        if not os.path.exists(dest) and not os.path.exists(zip_file_path):
            file_id = id_dict[k]
            download_file_from_google_drive(file_id, zip_file_path)

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

            print(f"Contents extracted to {dest}")

            os.remove(zip_file_path)

def save_obj(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def extract_values_from_pkl(folder_path, result_dict):

    # Recursively iterate through all subdirectories and files
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".pkl"):
                file_path = os.path.join(root, file_name)
                print(file_path)

                # Load the pickle file into a DataFrame
                df = pd.read_pickle(file_path)

                for _, row in df.iterrows():
                    key1 = row['mol_name']
                    key2 = file_name[:-4]

                    if key1 not in result_dict.keys():
                        result_dict[key1] = {}

                    result_dict[key1][key2] = file_path

    return result_dict
