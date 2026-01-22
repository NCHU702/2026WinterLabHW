# SW 淹水預測專案 (SW Flood Prediction Project)

本文件整合了資料前處理與模型訓練的完整流程說明。

## 1. 環境設置 (Environment Setup)

### 安裝相依套件 (Install Dependencies)
```bash
pip install -r requirements.txt
```

## 2. 專案結構 (File Structure)

本專案主要包含前處理腳本與模型訓練程式碼。

### 前處理相關檔案
```text
workspace_root/
  ├── inputs/
  │   ├── typhoon_hourly_rain_up_to_2023_OK.xlsx  (降雨資料)
  │   └── CWA_rain_targets_20251126_2323.csv      (測站資訊)
  ├── YL_SW/                                      (原始淹水模擬 .asc 檔案)
  ├── sw_data_all/                                (輸出資料夾)
  └── sw_data_preprocessing/                      (本資料夾)
      ├── config.py
      ├── process_floods.py
      └── process_rain.py
```

### 模型訓練相關檔案
```text
.
├── train.py            # 訓練
├── test.py             # 測試(輸出 CSV 檔)
├── dataload.py         # 資料讀取與 Dataset 定義
├── model.py            # UNet 模型
├── scan_dataset.py     # 掃描資料集並產生統計數據 (dataset_stats.json)
├── gen_rain_max.py     # (工具) 產生 rain_max.csv 
├── dataset_stats.json  # 儲存正規化所需的全域最大值 (由 scan_dataset.py 產生)
├── train_data          # 訓練集
├── test_data           # 測試集
├── val_data            # 驗證集
└── logs/               # TensorBoard 紀錄檔
```

---

## Part I: 資料前處理 (Data Preprocessing)

本階段用於將原始的淹水模擬數據與降雨測站數據，轉換為淹水預測模型所需的標準化 `sw_data_all` 格式。

### 設定 (Configuration)
若您的環境不同，請編輯 `config.py` 檢查路徑設定。
- `RAW_FLOOD_DIR`: 包含原始 `dm1d*.asc` 檔案的資料夾路徑 (例如 `YL_SW`)。
- `OUTPUT_DIR`: `sw_data_all` 的輸出路徑。

### 使用方法 (Usage)
#### 1. 處理淹水資料
此腳本將模擬輸出的 `.asc` 檔案轉換為 `.csv` 檔案，並將其整理至 `tX/flood` 資料夾中。同時也會提取 `metadata.txt`。

```bash
python process_floods.py
```

**注意**：在處理降雨資料之前必須先執行此步驟，因為元資料 (metadata) 被用作網格參考基準。

#### 2. 處理降雨資料
此腳本讀取降雨 Excel 檔案，將測站對應至座標，執行 IDW 內插法以符合淹水網格，並將結果儲存至 `tX/rain` 資料夾。

```bash
python process_rain.py
```

##### 對應關係
颱風名稱、來源資料夾與目標 ID (`tX`) 之間的對應關係定義於 `config.py` 中。

#### 3. 處理流程詳解 (Process Workflow)

執行 `process_flood.py`、`process_rain.py` 將自動完成以下所有步驟：

##### (1) 網格環境定義 (Metadata Analysis)
讀取網格參數，並計算出目標範圍的 TWD97 投影座標：
*   **Grid Dimensions**: 770 (cols) x 635 (rows)
*   **Cell Size**: 40 公尺
*   **TWD97 邊界範圍**:
    *   X: 162085.309 ~ 192885.309
    *   Y: 2600864.874 ~ 2626264.874
    *   (約對應 WGS84 經度 120.14°~120.44° / 緯度 23.51°~23.74°)

##### (2) 陸地遮罩建立 (Masking)
*   標記出數值為 `-999.999` (NODATA) 的區域。
*   確保後續產出的網格在相同位置（如非雲林縣範圍或海域）也是 NODATA。

##### (3) 測站定位與篩選 (Station Mapping & Filtering)
*   讀取 `inputs/typhoon_hourly_rain_up_to_2023_OK.xlsx` 中的資料表。
*   **定位**: 根據 `inputs/CWA_rain_targets_20251126_2323.csv` 提供的經緯度進行測站定位；若無測站資料，則參考該鄉鎮名稱之中心座標。
*   **座標轉換**: 將所有 WGS84 經緯度轉換為 TWD97 (EPSG:3826)。
*   **篩選**: 僅保留座標位於上述網格邊界內的測站。
*   **去重策略**: 若同一地點有多個測站名稱 (如 `斗六` 與 `斗六.1`)，保留數值最大者。
*   **結果**: 共有 **27 個有效測站** 被納入計算 (包含虎尾、土庫、臺西、四湖、北港等)。

##### (4) IDW 空間插值 (IDW Interpolation)
*   使用 **反距離加權法 (Inverse Distance Weighting, IDW)**。
*   針對每一個時間步長 (共 105 小時)，計算 770x635 網格上每個點的降雨數值。
*   將計算結果套用步驟 (2) 的遮罩，非計算區域填入 `-999.999`。

---

## Part II: 模型訓練與測試 (Model Training & Inference)

在資料前處理完成後，進入深度學習模型訓練階段。

### 1. 資料準備 (Data Preparation)

在開始訓練之前，必須先掃描資料集以取得全域最大值，用於 Min-Max Normalization。

1.  **準備資料結構**：
    確保 `train_data` 目錄下包含各個颱風事件的子資料夾，且每個事件內有 `rain` 和 `flood` 資料夾。

2.  **生成統計數據**：
    執行 `scan_dataset.py`，它會遍歷 `train_data` 資料夾，計算全域最大值並存檔。這一步對於正規化至關重要。

    ```bash
    python scan_dataset.py
    ```
    *   輸出：`dataset_stats.json` (包含 `max_rain` 與 `max_flood`)。

### 2. 模型訓練 (Training)

使用 `train.py` 進行模型訓練。

```bash
python train.py
```

您也可以透過指令參數自行調整訓練設定，例如：

```bash
python train.py --num_epochs 100 --scale 10.0 --train_batch_size 32
```

#### 可用參數說明 (Arguments)：

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

#### 訓練特性：
*   **正規化策略**：讀取 `dataset_stats.json` 進行全域 0~1 min-max normalization。
*   **Loss Function**：**Log-Weighted MSE**，針對淹水高值區域加權，解決樣本不平衡問題。
*   **監控**：支援 TensorBoard。

### 3. 測試與推論 (Inference)

訓練完成後，使用 `test.py` 對測試集進行推論。

```bash
python test.py
```

#### 推論參數說明：

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--test_data` | `test_data` | 測試資料集目錄名稱 |
| `--history_length` | `6` | 輸入歷史時間步長 |
| `--batch_size` | `8` | 測試批次大小 |
| `--mask_path` | `sw_mask.npy` | 淹水區域遮罩檔 |

**功能**：
1.  載入訓練好的模型權重 (預設讀取 `logs/` 下的 `best_model.pth`)。
2.  讀取 `dataset_stats.json` 確保正規化標準與訓練時一致。
3.  進行預測並反正規化 (Inverse Normalization)。
4.  輸出預測結果 CSV。
