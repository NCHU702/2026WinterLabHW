import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataload import TyphoonDataset, generate_flag_mask
from model import UNet
from train import load_stats
import argparse

def test(dir_name, history_length=6, batch_size=8, mask_path='sw_mask.npy'):
    # 參數設定
    '''
    dir_name: 測試資料夾名稱
    例如: 'test_data'
    '''
    # 這裡假設測試資料也在 test_data 資料夾下，實際情況請修改路徑
    root_dir = os.path.join(os.getcwd(), dir_name)
    history_length = 6
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 建立輸出目錄
    output_dir = f'{dir_name}_predictions'
    os.makedirs(output_dir, exist_ok=True)

    # 載入統計數據
    max_rain, max_flood = load_stats()
    print(f"Using normalization stats: Rain Max={max_rain}, Flood Max={max_flood}")

    # 載入資料集 (Mode='test')
    test_dataset = TyphoonDataset(root_dir, history_length, mode='test', max_rain=max_rain, max_flood=max_flood)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 載入模型結構
    model = UNet(in_channels=history_length + 1, out_channels=1).to(device)
    
    # 載入訓練好的權重
    # 假設最佳模型存在 logs 資料夾下，請確認路徑是否正確
    model_path = os.path.join(os.getcwd(), 'logs', 'best_model.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return

    # 生成 Flag Mask (需與訓練時一致)
    flag_mask = generate_flag_mask(mask_path=mask_path).to(device)

    model.eval()
    print("Start Testing...")
    global_idx = history_length
    with torch.no_grad():
        for i, rain_tensor in enumerate(tqdm(test_loader)):
            rain_tensor = rain_tensor.to(device)
            # test mode 下 Dataset只回傳 rain_tensor (B, T, H, W)
            
            # Handle Flag Mask with Batch Size
            B = rain_tensor.size(0)
            current_flag = flag_mask.expand(B, -1, -1, -1)

            # 推論
            output = model(rain_tensor, flag=current_flag) # (B, 1, H, W)
            
            # 對 Batch 中的每一筆資料進行處理
            output_np = output.cpu().numpy() # (B, 1, H, W)
            
            # 取得 mask 的 numpy (用於視覺化遮罩)
            flag_np = flag_mask[0, 0].cpu().numpy()

            for j in range(B):
                pred_flood = output_np[j, 0] # 取出單張圖 (H, W)
                
                # [反正規化] 還原真實深度 (公尺)
                pred_real = pred_flood * max_flood

                # 確保背景 (Mask=0) 為 -999，並儲存 CSV
                if pred_real.shape == flag_np.shape:
                    pred_real[flag_np == 0] = -999
                
                csv_path = os.path.join(output_dir, f'pred_{global_idx:04d}.csv')
                # 使用 pandas 儲存 CSV (無 header, 無 index)
                pd.DataFrame(pred_real).to_csv(csv_path, header=False, index=False)
                
                global_idx += 1

    print(f"Testing finished. Total {global_idx} files saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, default='test_data', help='測試資料集目錄')
    parser.add_argument('--history_length', type=int, default=6, help='歷史長度')
    parser.add_argument('--batch_size', type=int, default=8, help='測試批次大小')
    parser.add_argument('--mask_path', type=str, default=os.path.join(os.getcwd(), 'sw_mask.npy'))
    args = parser.parse_args()
    test(args.test_data, history_length=args.history_length, batch_size=args.batch_size, mask_path=args.mask_path)