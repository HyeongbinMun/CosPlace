import os
import cv2
from tqdm import tqdm
import argparse

class Images:
    def __init__(self, args):
        self.args = args
        self.oring_paph = args.origin_path
        self.resize_paph = args.resize_path
        self.img_type = args.img_type

        self.createfolder(self.resize_paph)

    def createfolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

    def resize(self, img_path, save_path):
        img_list = os.listdir(img_path)

        for img in tqdm(img_list):
            img_path = os.path.join(self.oring_paph, img)
            ori = cv2.imread(img_path, cv2.IMREAD_COLOR)
            res = cv2.resize(ori, dsize=(2000, 1000), interpolation=cv2.INTER_AREA)
            final_path = os.path.join(save_path, img)
            cv2.imwrite(final_path, res)

    def main(self):
        if self.img_type == 'files':
            self.resize(self.oring_paph, self.resize_paph)

        elif self.img_type == 'directory':
            img_dir_list = os.listdir(self.oring_paph)

            for img_dir in img_dir_list:
                img_path = os.path.join(self.oring_paph, img_dir)
                save_path = os.path.join(self.resize_paph, img_dir)
                self.createfolder(save_path)

                self.resize(img_path, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_path', type=str, default='/workspace/data/panorama/1796')
    parser.add_argument('--resize_path', type=str, default='/workspace/result/0113/panorama_result')
    parser.add_argument('--img_type', type=str, default='files', help='directory or files')
    args = parser.parse_args()

    img = Images(args)
    img.main()