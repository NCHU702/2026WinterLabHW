import pandas as pd
import numpy as np
import os
from pyproj import Transformer
import warnings
import config

def get_town_coords():
    # Known coordinates for specific station names or IDs found in the Excel file
    # Format: "Name": [Lat, Lon] (WGS84)
    return {
      "467480": [23.4958, 120.4334],
      "C0K280": [23.7032, 120.1983],
      "C0K300": [23.5746, 120.3024],
      "C0K330": [23.719183, 120.442036],
      "C0M530": [23.7602, 120.3539],
      "C1K230": [23.8387, 120.4324],
      "C1K250": [23.6391, 120.2263],
      "C1K260": [23.5828, 120.1837],
      "C1K270": [23.5755, 120.2458],
      "C1K310": [23.6765, 120.3927],
      "C1K320": [23.6483, 120.3130],
      "C1K340": [23.6763, 120.4792],
      "C1K350": [23.6455, 120.4300],
      "C1K380": [23.7632, 120.5057],
      "C1M470": [23.6391, 120.2263],
      "C1M500": [23.7082, 120.4339],
      "C1M510": [23.6765, 120.3927],
      "C1M520": [23.7011, 120.3111],
      "C1M540": [23.6744, 120.2526],
      "C1M560": [23.7536, 120.2526],
      "01J100": [23.800493, 120.464851],
      "01J930": [23.755254, 120.612491],
      "01J960": [23.7599, 120.6179],
      "01J970": [23.584167, 120.695634],
      "01K060": [23.701086, 120.311047],
      "01L360": [23.424513, 120.641716],
      "01L390": [23.475173, 120.620189],
      "01L480": [23.533049, 120.602587],
      "01L490": [23.530321, 120.520569],
      "01L910": [23.570458, 120.520964],
      "01M010": [23.587551, 120.40001],
      "O1J81": [23.5746, 120.3024],
      "麥寮": [23.7536, 120.2526],
      "崙背": [23.7602, 120.3539],
      "莿桐": [23.7632, 120.5057],
      "林內": [23.7599, 120.6179],
      "斗六": [23.7092, 120.5435],
      "斗南": [23.6763, 120.4792],
      "虎尾": [23.7082, 120.4339],
      "土庫": [23.6765, 120.3927],
      "元長": [23.6483, 120.3130],
      "褒忠": [23.7011, 120.3111],
      "東勢": [23.6744, 120.2526],
      "臺西": [23.7032, 120.1983],
      "四湖": [23.6391, 120.2263],
      "口湖": [23.5828, 120.1837],
      "水林": [23.5755, 120.2458],
      "北港": [23.5746, 120.3024],
      "大埤": [23.6455, 120.4300]
    }

def load_cwa_stations(csv_path):
    station_map = {}
    if not os.path.exists(csv_path):
        print(f"Warning: Station CSV not found at {csv_path}")
        return station_map
    
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            sid = str(row['StationId']).strip()
            sname = str(row['StationName']).strip()
            lat = float(row['Latitude_WGS84'])
            lon = float(row['Longitude_WGS84'])
            
            station_map[sid] = [lat, lon]
            station_map[sname] = [lat, lon]
    except Exception as e:
        print(f"Error loading CWA stations: {e}")
    
    return station_map

def main():
    warnings.filterwarnings('ignore')
    print("--- Starting Rain Grid Generation (IDW) ---")

    # Paths from config
    # Reference metadata from t5 (Morakot)
    ref_dir = os.path.join(config.OUTPUT_DIR, "t5", "flood")
    metadata_path = os.path.join(ref_dir, "metadata.txt")
    # Use the first flood CSV as mask reference (assumes process_floods.py ran for t5)
    t5_ref_csv = os.path.join(ref_dir, "dm1d0000.csv")
    
    if not os.path.exists(metadata_path) or not os.path.exists(t5_ref_csv):
        print(f"ERROR: Reference metadata/mask not found in {ref_dir}.")
        print("Please run 'process_floods.py' first.")
        return

    # 1. Load Metadata
    print("Reading Metadata...")
    meta = {}
    with open(metadata_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.split()
            if len(parts) >= 2:
                meta[parts[0]] = float(parts[1])

    ncols = int(meta['ncols'])
    nrows = int(meta['nrows'])
    xll = meta['xllcorner']
    yll = meta['yllcorner']
    cellsize = meta['cellsize']
    nodata = meta['NODATA_value']

    xur = xll + ncols * cellsize
    yur = yll + nrows * cellsize
    print(f"Grid TWD97 Bounds: X[{xll:.1f}, {xur:.1f}], Y[{yll:.1f}, {yur:.1f}]")

    # 2. Load Mask
    print("Loading Mask from t5 flood data...")
    # Read without header
    t5_df = pd.read_csv(t5_ref_csv, header=None)
    t5_data = t5_df.values
    # Mask True where value is NODATA (or close to it)
    mask = np.isclose(t5_data, nodata, atol=1e-3)
    mask_flat = mask.ravel()

    # 3. Setup Station Coordinates
    known_coords = get_town_coords()
    cwa_coords = load_cwa_stations(config.STATIONS_CSV_PATH)
    transformer = Transformer.from_crs("epsg:4326", "epsg:3826", always_xy=True)

    # 4. Precompute Grid Points for IDW
    x_coords = np.linspace(xll + cellsize/2, xur - cellsize/2, ncols)
    # Usually yllcorner is Bottom-Left. 
    # If the file is stored top-to-bottom (standard for some ASC), Y decreases.
    # However, usually we align with how flood data was read.
    # If flood data was read via pd.read_csv without manipulation, it matches row-major.
    # If standard ASC: Row 0 is Y_max.
    y_coords = np.linspace(yll + nrows*cellsize - cellsize/2, yll + cellsize/2, nrows)
    
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))

    # 5. Process Excel Sheets
    print(f"Reading Excel File: {config.RAIN_EXCEL_PATH}")
    xls = pd.ExcelFile(config.RAIN_EXCEL_PATH)
    sheets = xls.sheet_names
    
    print(f"Found {len(sheets)} sheets.")

    for sheet_name in sheets:
        if "工作表" in sheet_name or "Sheet" in sheet_name:
            continue
        
        # Determine tX output folder
        # We need to match sheet_name to config.NAME_TO_ID_MAPPING key
        # Sheet names might be "2009_莫拉克"
        # config keys are "2009_莫拉克"
        
        # Fuzzy match or direct match
        target_id = config.NAME_TO_ID_MAPPING.get(sheet_name)
        if not target_id:
            print(f"Skipping {sheet_name}: No mapping found in config.")
            continue
            
        out_dir = os.path.join(config.OUTPUT_DIR, target_id, "rain")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        print(f"\n=== Processing {sheet_name} -> {target_id} ===")
        
        df_sheet = pd.read_excel(config.RAIN_EXCEL_PATH, sheet_name=sheet_name)
        
        # Identify Stations
        stations = []
        data_cols = []
        
        for col in df_sheet.columns:
            if col == 'DataTime': continue
            
            col_str = str(col)
            base_name = col_str.split('.')[0].strip()
            
            coords = None
            if col_str in known_coords:
                coords = known_coords[col_str]
            elif base_name in known_coords:
                coords = known_coords[base_name]
            elif col_str in cwa_coords:
                coords = cwa_coords[col_str]
            elif base_name in cwa_coords:
                coords = cwa_coords[base_name]
            
            if coords:
                # Transform to TWD97
                sx, sy = transformer.transform(coords[1], coords[0]) # lon, lat
                
                # Filter bounds
                if (sx >= xll and sx <= xur and sy >= yll and sy <= yur):
                    stations.append([sx, sy])
                    data_cols.append(col)
        
        if not stations:
            print(f"No valid stations found inside bounds for {sheet_name}")
            continue
            
        station_points = np.array(stations)
        # IDW Weights
        dif = grid_points[:, np.newaxis, :] - station_points[np.newaxis, :, :]
        dist_sq = np.sum(dif ** 2, axis=2)
        epsilon = 1e-6
        dist_sq[dist_sq < epsilon] = epsilon
        weights = 1.0 / dist_sq
        sum_weights = np.sum(weights, axis=1)
        
        # Process Time Steps
        station_vals = df_sheet[data_cols].values # shape (T, N_stations)
        
        for t in range(len(df_sheet)):
            z_vals = station_vals[t, :]
            
            # Interpolate
            weighted_sum = np.dot(weights, z_vals)
            z_interp = weighted_sum / sum_weights
            
            # Mask
            z_interp[mask_flat] = nodata
            z_grid = z_interp.reshape(nrows, ncols)
            
            # Save
            # Format: '2009_莫拉克_0000.csv'
            fname = f"{sheet_name}_{t:04d}.csv"
            fpath = os.path.join(out_dir, fname)
            pd.DataFrame(z_grid).to_csv(fpath, header=False, index=False)
            
            if t % 24 == 0:
                print(f"Processed {t}/{len(df_sheet)}", end='\r')
        
        print(f"Finished {sheet_name}")

if __name__ == "__main__":
    main()
