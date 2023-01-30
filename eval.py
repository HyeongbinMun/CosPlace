
import sys
import torch
import logging
import multiprocessing
import time
import parser
from datetime import datetime
torch.backends.cudnn.benchmark= True  # Provides a speedup

from test import generate_matching_list
from utils import commons
from models.cosplays import cosplace
from datasets.test_dataset import InferDataset
from utils.recall import Predict


if __name__ == '__main__':
    args = parser.parse_arguments(is_training=False)
    start_time = datetime.now()
    output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(args.seed)
    commons.setup_logging(output_folder, console="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")

    #### Model
    model = cosplace.GeoLocalizationNet(args.backbone, args.fc_output_dim)

    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

    if args.resume_model != None:
        logging.info(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        model.load_state_dict(model_state_dict)
    else:
        logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                     "Evaluation will be computed using randomly initialized weights.")

    model = model.to(args.device)
    prec = Predict(args)
    db = args.db_dir
    query = args.query_dir

    start_load_time = time.time()
    test_ds = InferDataset(dataset_folder=db, queryset_folder=query)
    end_load_time = time.time()
    total_load_time = end_load_time - start_load_time
    logging.info(f"data load time : {total_load_time}")

    start_total_time = time.time()
    generate_matching_list(args, test_ds, model)
    end_total_time = time.time()
    total_process_time = end_total_time - start_total_time
    logging.info(f"query  search time : {total_process_time}")

    if args.gt_dir:
        prec.main()
