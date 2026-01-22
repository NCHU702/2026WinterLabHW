import torch
import torch.nn as nn
import torch.optim as optim
from dataload import TyphoonDataset, generate_flag_mask
from model import UNet
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataload import load_stats
import argparse


def visualize_sample(val_set, epoch, flag_mask, writer, sample_idx=9, max_val=1.0):
    """
    可視化驗證集中的單筆樣本，並將結果儲存及記錄到 TensorBoard。
    Args:
        max_val (float): 用於反正規化
    """
    rain_sample, flood_sample = val_set[sample_idx]
    # 增加 Batch 維度並移至 Device
    rain_tensor = rain_sample.unsqueeze(0).to(device) # (1, T, H, W)
    flood_target = flood_sample.unsqueeze(0).to(device) #1 (1, 1, H, W)
    # 預測
    output = model(rain_tensor, flag=flag_mask)
    
    # 轉回 CPU numpy 用於繪圖
    target_flood = flood_target.squeeze().cpu().numpy()
    pred_flood = output.squeeze().cpu().numpy()

    # --- 反正規化 ---
    target_flood = target_flood * max_val
    pred_flood = pred_flood * max_val
    # ---------------
    
    # 將 mask 為 0 的區域設為 NaN，繪圖時會自動忽略 (變白)
    mask_np = flag_mask.cpu().numpy().squeeze()
    
    # 確保 mask 形狀正確
    if mask_np.shape == target_flood.shape:
        target_flood[mask_np == 0] = np.nan
        pred_flood[mask_np == 0] = np.nan

    # 繪圖
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 使用 vmax=max_flood 確保色階一致
    im1 = axes[0].imshow(target_flood, cmap='Oranges', vmin=0, vmax=max_flood)
    axes[0].set_title(f'Target Flood (t)')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(pred_flood, cmap='Oranges', vmin=0, vmax=max_flood)
    axes[1].set_title(f'Pred Flood (Epoch {epoch})')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    
    # 儲存圖片並加到 TensorBoard
    save_path = os.path.join(writer.log_dir, 'vis_epoch',f'vis_epoch_{epoch}.png')
    if os.path.exists(os.path.dirname(save_path)) is False:
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    writer.add_figure(f'Validation Sample {sample_idx}', fig, epoch)
    plt.close(fig)
    print(f"Saved visualization for epoch {epoch} at {save_path}")

if __name__ == "__main__":
    # 輸入參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root_dir', type=str, default=os.path.join(os.getcwd(), 'train_data'))
    parser.add_argument('--val_root_dir', type=str, default=os.path.join(os.getcwd(), 'val_data'))
    parser.add_argument('--mask_path', type=str, default=os.path.join(os.getcwd(), 'sw_mask.npy'))
    parser.add_argument('--history_length', type=int, default=6)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--scale', type=float, default=10.0)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # 設定參數
    train_root_dir = args.train_root_dir  # 訓練資料集根目錄
    val_root_dir = args.val_root_dir      # 驗證資料集根目錄
    mask_path = args.mask_path  # SW Mask 檔案路徑
    history_length = args.history_length # 輸入的時間長度
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    scale = args.scale # 影響loss reweighting  的比例因子
    num_workers = args.num_workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 從訓練集載入統計數據
    max_rain, max_flood = load_stats()
    print(f"Using normalization stats: Rain Max={max_rain}, Flood Max={max_flood}")

    # 建立資料集和資料加載器
    train_set = TyphoonDataset(train_root_dir, history_length, mode='train', max_rain=max_rain, max_flood=max_flood)
    val_set = TyphoonDataset(val_root_dir, history_length, mode='val', max_rain=max_rain, max_flood=max_flood)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    # 建立模型、損失函數和優化器
    model = UNet(in_channels=history_length + 1, out_channels=1).to(device)
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # TensorBoard 記錄器
    if not os.path.exists(os.path.join(os.getcwd(), 'logs')):
        os.makedirs(os.path.join(os.getcwd(), 'logs'))
    writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), 'logs'))
    # 生成 binary flag mask
    # 在 loop 裡自己做 expand
    flag_mask = generate_flag_mask(mask_path).to(device)  # (1, 1, H, W)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # 訓練迴圈
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        current_flag = []
        for rain_tensor, flood_tensor in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs-1} [Train]'):
            rain_tensor = rain_tensor.to(device)
            flood_tensor = flood_tensor.to(device)
            #檢查 batch size 是否匹配 flag mask
            B = rain_tensor.size(0)
            
            # flag_mask 現在是 (1, 1, H, W)，需要 expand 它到目前的 batch size
            current_flag = flag_mask.expand(B, -1, -1, -1)

            # 前向傳播
            outputs = model(rain_tensor, flag=current_flag)

            # 計算損失
            # 1. MSE Loss (Pixel-wise)
            loss_map = criterion(outputs, flood_tensor) # (pred-target)^2
            
            # [Weighted Loss](對數加權)
            # 權重 W = 1 + log1p(Target) * scale
            # 讓淹水越深的地方，權重越大。Target 為 0 (沒淹水) 時，log1p(0)=0 -> W=1 (正常權重)
            log_weight = 1 + torch.log1p(flood_tensor * current_flag) * scale
            
            # 結合權重
            loss_map = loss_map * log_weight

            # 乘上 flag mask (只看有效區域)
            loss = (loss_map * current_flag).sum() / (current_flag.sum() + 1e-6)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * rain_tensor.size(0)

        train_loss /= len(train_set)
        train_losses.append(train_loss)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rain_tensor, flood_tensor in tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs-1} [Val]'):
                rain_tensor = rain_tensor.to(device)
                flood_tensor = flood_tensor.to(device)
                
                # 處理 Val Phase 的 Mask Batch Size 問題
                B = rain_tensor.size(0)
                current_flag = flag_mask.expand(B, -1, -1, -1)

                outputs = model(rain_tensor, flag=current_flag)

                loss_map = criterion(outputs, flood_tensor)
                log_weight = 1 + torch.log1p(flood_tensor * current_flag) * scale
                loss_map = loss_map * log_weight
                loss = (loss_map * current_flag).sum() / (current_flag.sum() + 1e-6)
               
                val_loss += loss.item() * rain_tensor.size(0)
            if epoch % 5 == 0:
                visualize_sample(val_set, epoch, flag_mask, writer, sample_idx=9, max_flood=max_flood)

        val_loss /= len(val_set)
        val_losses.append(val_loss)
        writer.add_scalar('Loss/val', val_loss, epoch)

        print(f'Epoch [{epoch}/{num_epochs-1}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_model.pth'))
            
       
    # --- Loss Visualization ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_fig_save_path = os.path.join(writer.log_dir, 'loss_curve.png')
    plt.savefig(loss_fig_save_path)
    print(f"Loss curve saved as {loss_fig_save_path}")

