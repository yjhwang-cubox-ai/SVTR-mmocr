from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import cv2
import json
from dictionary import Dictionary
from config import Config

class SvtrDataset(Dataset):
    def __init__(self):
        self.config = Config()
        self.dictionary = Dictionary(dict_file=self.config.global_config['character_dict_path'])
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        img, text = self._get_img_text(index)
        img = self.transform(img)
        target = self.dictionary.str2idx(text)

        return img, text, target
    
    def _get_img_text(self, index):
        raise NotImplementedError
    

class TNGoDataset(SvtrDataset):
    def __init__(self, dataset_json: list, mode:str):
        super().__init__()
        self.dataset_json = dataset_json
        self.data_list = []

        for json_file in dataset_json:
            # dir_path = os.path.dirname(json_file)
            with open(json_file, 'r', encoding='utf-8') as f:
                data_info = json.load(f)['data_list']
            for data in data_info:
                data['img_path'] = os.path.join(os.path.dirname(json_file), data['img_path'])
            self.data_list.extend(data_info)
        
        self.mode = mode

    def __len__(self):
        return len(self.data_list)

    def _get_img_text(self, index):
        img_path = self.data_list[index]['img_path']
        img = cv2.imread(img_path)
        text = self.data_list[index]['instances'][0]['text']
        return img, text


if __name__ == '__main__':
    dataset_json = ['/data/TNGoDataset/1_TNGo1/annotation.json',
                    '/data/TNGoDataset/3_TNGo3/annotation.json']
    dataset = TNGoDataset(dataset_json, mode='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    for batch in dataloader:
        print(batch)


    output = dataset._print()
    print(output)