import os
import shutil

class Predict:
    def __init__(self, args):
        self.args = args
        self.gt_path = args.gt_dir
        self.result_path = os.path.join(args.result_path, 'recall_img')
        self.result_txt = os.path.join(args.result_path, args.result_txt_name)

        self.createfolder(args.result_path)
        self.createfolder(self.result_path)

    def createfolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

    def file_read(self, path):
        tmp = []
        f = open(path, 'r')
        while True:
            line = f.readline()
            if not line: break
            result = line.split()
            result[-1] = result[-1].strip()
            tmp.append(result)
        f.close()
        return tmp

    def copy_file(self, path, save_path):
        for el in path:
            files = el[0].split('/')
            image_name, _ = files[-1].split('.')
            image_dir = os.path.join(save_path, image_name)
            self.createfolder(image_dir)

            rename_image = os.path.join(image_dir, files[-1])
            shutil.copyfile(el[0], rename_image)
            for i in range(1, len(el)):
                rename_image = os.path.join(image_dir, f'{i}.jpg')
                shutil.copyfile(el[i], rename_image)

    def gt_save(self, data):
        tmp = {}
        for gt in data:
            el = gt.split('@')
            key = el[0]
            tmp[key] = el[1]
        return tmp

    def recall_back(self, result, gt, num):
        recall = 0
        for el in result:
            files = el[0].split('/')
            img = files[-1].split('@')
            key = img[0]

            value = gt.get(key)
            for i in range(1, num):
                files = el[i].split('/')
                if value == files[-1]:
                    recall += 1

        total_recall = recall / len(result) * 100
        return total_recall

    def main(self):
        cos_result_list = self.file_read(self.result_txt)
        if len(os.listdir(self.result_path)) == 0:
            self.copy_file(cos_result_list, self.result_path)

        gt_list = os.listdir(self.gt_path)
        gt_value = self.gt_save(gt_list)

        panorama_recall = self.recall_back(cos_result_list, gt_value, 11)
        print(f'recall@10 : {panorama_recall:.2f}\n')

        panorama_recall = self.recall_back(cos_result_list, gt_value, 6)
        print(f'recall@5 : {panorama_recall:.2f}\n')

        panorama_recall = self.recall_back(cos_result_list, gt_value, 2)
        print(f'recall@1 : {panorama_recall:.2f}\n')

