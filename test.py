import torch, faiss, logging, os
import multiprocessing, time
import numpy as np
from tqdm import tqdm
from utils.recall import Predict
from utils.commons import load_settings
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from models.cosplays import cosplace
from datasets.test_dataset import InferDataset, SingleDataset

PARAMS_PATH = '/workspace/cfg/config/params.yaml'

class Cosplace:
    def __init__(self, params=load_settings(PARAMS_PATH)):
        self.params = params
        self.path = params['model']['data']
        self.cosplace = params['model']['cosplace']
        self.model = self.load_model(self.cosplace)
        self.recall_num = self.cosplace['recall_value']

        # Compute R@1, R@5, R@10, R@20
        self.recall_value = [self.recall_num]
        self.createfolder(self.path['save'])

    def createfolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

    def load_model(self, args):
        model = cosplace.GeoLocalizationNet(args['backbone'], args['fc_output_dim'])
        logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

        if args['model_path'] != None:
            logging.info(f"Loading model from {args['model_path']}")
            model_state_dict = torch.load(args['model_path'])
            model.load_state_dict(model_state_dict)
        else:
            logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                         "Evaluation will be computed using randomly initialized weights.")
        model = model.to(args['device'])

        return model

    def inference_by_images(self, query):
        results = []
        start_load_time = time.time()
        if '.' in query:
            eval_ds = SingleDataset(dataset_folder=self.path['db'], queryset_image=query)
        else:
            eval_ds = InferDataset(dataset_folder=self.path['db'], queryset_folder=query)
        end_load_time = time.time()
        total_load_time = end_load_time - start_load_time
        logging.info(f"data load time : {total_load_time}")

        """Compute descriptors of the given dataset and compute the recalls."""
        self.model = self.model.eval()

        with torch.no_grad():
            logging.debug("Extracting database descriptors for evaluation/testing")
            database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
            database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=self.cosplace['num_workers'],
                                             batch_size=self.cosplace['batch_size'], pin_memory=(self.cosplace['device'] == "cuda"))
            all_descriptors = np.empty((len(eval_ds), self.cosplace['fc_output_dim']), dtype="float32")
            if os.path.isfile(self.path['feature']):
                all_descriptors[:eval_ds.database_num] = np.load(self.path['feature'])
            else:
                for images, indices in tqdm(database_dataloader, ncols=100):
                    descriptors = self.model(images.to(self.cosplace['device']))
                    descriptors = descriptors.cpu().numpy()
                    all_descriptors[indices.numpy(), :] = descriptors

            logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
            queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)))
            queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=self.cosplace['num_workers'],
                                            batch_size=self.cosplace['batch_size'], pin_memory=(self.cosplace['device'] == "cuda"))
            for images, indices in tqdm(queries_dataloader, ncols=100):
                descriptors = self.model(images.to(self.cosplace['device']))
                descriptors = descriptors.cpu().numpy()
                all_descriptors[indices.numpy(), :] = descriptors

        queries_descriptors = all_descriptors[eval_ds.database_num:]
        database_descriptors = all_descriptors[:eval_ds.database_num]
        if not os.path.isfile(self.path['feature']):
            np.save(self.path['feature'], database_descriptors)

        # Use a kNN to find predictions
        faiss_index = faiss.IndexFlatL2(self.cosplace['fc_output_dim'])
        faiss_index.add(database_descriptors)
        del database_descriptors, all_descriptors

        logging.debug("Calculating recalls")
        _, predictions = faiss_index.search(queries_descriptors, max(self.recall_value))

        result_txt_path = os.path.join(self.path['save'], self.path['save_name'])
        matching_file = open(result_txt_path, 'a')
        #### For each query, check if the predictions are correct
        data_paths = eval_ds.get_datapaths()
        for query_index, preds in enumerate(predictions):
            for i, n in enumerate(self.recall_value):
                db_candiate = np.array(data_paths)[preds[:10]].tolist()
                query_path = data_paths[eval_ds.database_num + query_index]
                dt_str = ' '.join(db_candiate)
                results = dt_str
                matching_str = f'{query_path} {dt_str}\n'
                matching_file.write(matching_str)
        matching_file.close()

        return results

