import torch, faiss, logging, os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

def generate_matching_list(args, eval_ds, model):
    """Compute descriptors of the given dataset and compute the recalls."""

    # Compute R@1, R@5, R@10, R@20
    RECALL_VALUES = [args.recall_value]
    model = model.eval()

    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
        if os.path.isfile(args.feature_path):
            all_descriptors[:eval_ds.database_num] = np.load(args.feature_path)
        else:
            for images, indices in tqdm(database_dataloader, ncols=100):
                descriptors = model(images.to(args.device))
                descriptors = descriptors.cpu().numpy()
                all_descriptors[indices.numpy(), :] = descriptors

        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_subset_ds = Subset(eval_ds,
                                   list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[eval_ds.database_num:]
    database_descriptors = all_descriptors[:eval_ds.database_num]
    if not os.path.isfile(args.feature_path):
        np.save(args.feature_path, database_descriptors)

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))

    result_txt_path = os.path.join(args.result_path, args.result_txt_name)
    matching_file = open(result_txt_path, 'a')
    #### For each query, check if the predictions are correct
    data_paths = eval_ds.get_datapaths()
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            db_candiate = np.array(data_paths)[preds[:10]].tolist()
            query_path = data_paths[eval_ds.database_num + query_index]
            dt_str = ' '.join(db_candiate)
            matching_str = f'{query_path} {dt_str}\n'
            matching_file.write(matching_str)
    matching_file.close()



