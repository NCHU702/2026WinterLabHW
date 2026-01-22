## 1. 環境設置 (Environment Setup)

# 安裝相依套件
pip install -r requirements.txt
```

## 2. 專案結構 (File Structure)

```text
.
├── train.py            # 訓練
├── test.py             # 測試(輸出 CSV 檔)
├── dataload.py         # 資料讀取與 Dataset 定義
├── model.py            # UNet 模型
├── scan_dataset.py     # 掃描資料集並產生統計數據 (dataset_stats.json)
├── gen_rain_max.py     # (工具) 產生 rain_max.csv 
├── dataset_stats.json  # 儲存正規化所需的全域最大值 (由 scan_dataset.py 產生)
├── train_data # 訓練集
├── test_data  # 測試集
├── val_data   # 驗證集
│  
└── logs/               # TensorBoard 紀錄檔
```

## 3. 資料準備 (Data Preparation)

在開始訓練之前，必須先掃描資料集以取得全域最大值，用於 Min-Max Normalization。

1. **準備資料結構**：
   確保 `train_data` 目錄下包含各個颱風事件的子資料夾，且每個事件內有 `rain` 和 `flood` 資料夾。

2. **前處理 (Optional)**：
   若資料夾中尚未有 `*_max.csv` 統計檔，可執行gen_rain_max.py。

3. **生成統計數據**：
   執行 `scan_dataset.py`，它會遍歷train_data資料夾尋找 `rain_max.csv` 與 `flood_max.csv`，計算全域最大值並存檔。

   ```bash
   python scan_dataset.py
   ```
   * 輸出：`dataset_stats.json` (包含 `max_rain` 與 `max_flood`)。

## 4. 模型訓練 (Training)

使用 `train.py` 進行模型訓練。

```bash
python train.py
```

您也可以透過指令參數自行調整訓練設定，例如：

```bash
python train.py --num_epochs 100 --scale 10.0 --train_batch_size 32
```

### 可用參數說明：

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--train_root_dir` | `train_data` | 訓練資料集目錄路徑 |
| `--val_root_dir` | `val_data` | 驗證資料集目錄路徑 |
| `--mask_path` | `sw_mask.npy` | 淹水區域遮罩檔 (只計算有效區域的 Loss) |
| `--history_length` | `6` | 輸入歷史時間步長|
| `--train_batch_size`| `16` | 訓練時的 Batch Size |
| `--val_batch_size` | `8` | 驗證時的 Batch Size |
| `--num_epochs` | `50` | 總訓練回合數 |
| `--learning_rate` | `0.0001` | 學習率 (Learning Rate) |
| `--scale` | `10.0` | Log-Weighted Loss 的加權係數 (數值越大，對淹水區越敏感) |
| `--num_workers` | `4` | DataLoader 多工讀取執行緒數 |

### 訓練特性：
* **正規化策略**：讀取 `dataset_stats.json` 進行全域 0~1 min-max normalization。
* **Loss Function**：**Log-Weighted MSE**
 針對淹水數值較高區域給予較大權重 (預設scale=10.0)，目的在於想要解決樣本不平衡導致模型傾向預測 0 的問題。
* **監控**：支援 TensorBoard，可即時查看 Loss 曲線與預測影像。


## 5. 測試與推論 (Inference)

訓練完成後，使用 `test.py` 對測試集進行推論。

```bash
python test.py
```

### 可用參數說明：

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--test_data` | `test_data` | 測試資料集目錄名稱 |
| `--history_length` | `6` | 輸入歷史時間步長 |
| `--batch_size` | `8` | 測試批次大小 |
| `--mask_path` | `sw_mask.npy` | 淹水區域遮罩檔 |

* **功能**：
    1. 載入訓練好的模型權重 (預設讀取 `logs/` 下的 `best_model.pth`)。
    2. 讀取 `dataset_stats.json` 確保正規化標準與訓練時一致。
    3. 進行預測並反正規化 (Inverse Normalization)。
    4. 輸出預測結果 CSV 

## 6. 模型架構

* **Model**: UNet
* **Input**: 多個時間步的累積雨量圖+mask (Batch, Time + 1, H, W)
* **Output**: 單一時間步的淹水深度圖。(Batch, 1, H, W)
