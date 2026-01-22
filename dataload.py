import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import json

def load_stats(json_path='dataset_stats.json'):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Stats file '{json_path}' not found. Please run scan_dataset.py first.")
    
    try:
        with open(json_path, 'r') as f:
            stats = json.load(f)
        
        # 檢查關鍵字是否存在
        if 'max_rain' not in stats or 'max_flood' not in stats:
            raise KeyError(f"Missing keys 'max_rain' or 'max_flood' in {json_path}")
            
        return float(stats['max_rain']), float(stats['max_flood'])
        
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from {json_path}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading stats: {e}")

class TyphoonDataset(Dataset):
    def __init__(self, root_dir, history_length=6, mode='train', max_rain=None, max_flood=None):
        """
        Args:
            root_dir (str): 資料集的根目錄
            history_length (int): 輸入的時間長度
            mode (str): 資料集模式(test無target)
            max_rain (float): 訓練資料集的全域最大雨量 
            max_flood (float): 訓練資料集的全域最大淹水深度 
        """
        self.root_dir = root_dir
        self.history_length = history_length
        try :
            assert max_rain is not None and max_flood is not None, "max_rain and max_flood must be provided."
        except AssertionError as e:
            raise RuntimeError(f"Initialization error: {e}")
        self.max_rain = max_rain
        self.max_flood = max_flood
        self.samples = []
        self.event_list = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.mode = mode
        # 預處理：建立所有合法的 (event, target_time_step) 索引
        # 這樣可以確保 __getitem__ 時不會跨越不同颱風事件
        self.event_files = {} # 儲存每個事件的檔案列表，避免寫死檔名格式
        self._prepare_indices()

    def _get_sorted_files(self, dir_path):
        """取得目錄下排序好的資料檔案 (支援 csv, npy)"""
        if not os.path.exists(dir_path):
            return []
        # 過濾非資料檔案並過濾掉檔名中有max的檔案
        files = [f for f in os.listdir(dir_path) 
                 if f.lower().endswith(('.csv')) and not f.startswith('.') and 'max' not in f]
        files.sort() # 確保時間順序
        return files

    def _load_file(self, path):
        """根據副檔名讀取檔案"""
        if path.endswith('.csv'):
            return pd.read_csv(path, header=None).values
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def _prepare_indices(self):
        for event_id in self.event_list:
            rain_dir = os.path.join(self.root_dir, event_id, 'rain')
            flood_dir = os.path.join(self.root_dir, event_id, 'flood')
            
            rain_files = self._get_sorted_files(rain_dir)
            flood_files = self._get_sorted_files(flood_dir)
            
            # 使用列表來管理檔案，不再依賴檔名格式
            self.event_files[event_id] = {
                'rain': rain_files,
                'flood': flood_files
            }
            
            num_timesteps = len(rain_files)
            
            # 假設時間從 0 開始，若 t=6 (第7小時)，則 input 為 0,1,2,3,4,5 (共6個)
            for t in range(self.history_length, num_timesteps):
                self.samples.append((event_id, t))
            print(f"Prepared {num_timesteps - self.history_length} samples for event {event_id}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        event_id, target_t = self.samples[index]
        # 計算輸入序列的時間範圍: [target_t - history_length, target_t - 1]
        input_indices = list(range(target_t - self.history_length, target_t))
        
        rain_data = []
        rain_files = self.event_files[event_id]['rain']
        
        for t in input_indices:
            # 直接從列表中取得對應時間的檔案名稱
            filename = rain_files[t]
            rain_path = os.path.join(self.root_dir, event_id, 'rain', filename)
            frame = self._load_file(rain_path)
            rain_data.append(frame)
        
        rain_array = np.stack(rain_data).astype(np.float32)

        # [Rain Normalization] Val = Val / Max
        rain_array = rain_array / self.max_rain
        # 只做下限 Clip, 上限不做 (保留外推能力)
        # 在這邊-999都會被砍成0
        rain_array = np.clip(rain_array, 0.0, None)

        rain_tensor = torch.from_numpy(rain_array)

        if (self.mode == 'train') or (self.mode == 'val'):
            flood_files = self.event_files[event_id]['flood']
            # 注意: target_t 對應的 flood 檔案應該是 flood_files[target_t]
            # 前提是 rain 和 flood 的檔案數量與順序是一致的
            if target_t < len(flood_files):
                filename = flood_files[target_t]
                flood_path = os.path.join(self.root_dir, event_id, 'flood', filename)
                flood_array = self._load_file(flood_path).astype(np.float32)

                # [Flood Normalization] Val = Val / Max
                flood_array = flood_array / self.max_flood
                flood_array = np.clip(flood_array, 0.0, None)#在這邊-999都會被砍成0

                flood_tensor = torch.from_numpy(flood_array).unsqueeze(0) # (H, W) -> (1, H, W)
                return rain_tensor, flood_tensor
            else:
                 # Fallback handle potentially missing flood file? Or raise error
                 raise IndexError(f"Flood file index {target_t} out of range for event {event_id}")
        else:
            return rain_tensor

def generate_flag_mask(mask_path='sw_mask.npy'):
    """
    讀取 SW Mask 後回傳 Tensor。
    Args:
        mask_path (str): mask 檔案路徑，預設為 'sw_mask.npy'
    Returns:
        torch.Tensor: 形狀為 (1, 1, H, W) 的 binary flag mask
    """
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
    else:
        raise FileNotFoundError(f"SW Mask file not found at {mask_path}. Please ensure the file exists.")
    # 將 mask 轉為 torch.Tensor 並擴展 batch 維度
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
    return mask_tensor

if __name__ == "__main__":
    # 測試資料集
    root_dir = os.path.join(os.getcwd(), 'train_data') 
    max_rain, max_flood = load_stats()
    dataset = TyphoonDataset(root_dir, history_length=6, mode='train', max_rain=max_rain, max_flood=max_flood)
    print(f"Dataset size: {len(dataset)}")
    rain_tensor, flood_tensor = dataset[90]
    print(f"Rain tensor shape: {rain_tensor.shape}")   # 預期 (6, H, W)
    print(f"Flood tensor shape: {flood_tensor.shape}") # 預期 (1, H, W)
    print(rain_tensor.max())

