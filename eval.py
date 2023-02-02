import sys
import torch
import logging
import time
import argparse
from datetime import datetime
torch.backends.cudnn.benchmark= True  # Provides a speedup

from test import Cosplace
from utils import commons
from utils.recall import Predict
from utils.commons import load_settings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', type=str, default='/workspace/data/sample/resize_queries')
    parser.add_argument('--params', type=str, default='/workspace/cfg/config/params.yaml')
    args = parser.parse_args()

    option = parser.parse_known_args()[0]
    params_path = option.params
    params = load_settings(params_path)
    data = params['model']['data']
    cosplace = params['model']['cosplace']

    start_time = datetime.now()
    output_folder = f"logs/{data['save']}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(cosplace['seed'])
    commons.setup_logging(output_folder, console="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")

    cos = Cosplace(params)
    prec = Predict(params)
    query_path = args.query_path

    start_total_time = time.time()
    predictions = cos.inference_by_images(query_path)
    end_total_time = time.time()
    total_process_time = end_total_time - start_total_time
    logging.info(f"query  search time : {total_process_time}")
    logging.info(f"query  result : {predictions}")

    prec.main()
