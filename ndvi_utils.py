import os
import numpy as np
import rasterio
import pandas as pd
from datetime import datetime
import ee
import rasterio


# === 🧩 EARTH ENGINE INIT ===
SERVICE_ACCOUNT = 'earthengine-sa@internship-450608.iam.gserviceaccount.com'
SERVICE_ACCOUNT_KEY_PATH = '/home/user/airflow/internship-450608-612eec62b310.json'

def init_ee():
    """
    Initializes the Google Earth Engine using a service account.

    Inputs:
        None (uses global SERVICE_ACCOUNT and SERVICE_ACCOUNT_KEY_PATH)

    Output:
        None, but prints a confirmation message if initialization is successful.
    """
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, SERVICE_ACCOUNT_KEY_PATH)
    ee.Initialize(credentials)
    print("✅ Earth Engine initialized with service account.")

# =====================================================================

import os
import rasterio

def compute_difference_tif(tif_before_path, tif_after_path, OUTPUT):
    """
    Computes the pixel-wise difference between two TIFF images and saves the result.

    Inputs:
        tif_before_path (str): Path to the 'before' TIFF image.
        tif_after_path (str): Path to the 'after' TIFF image.
        OUTPUT (str): Directory path where the resulting difference image will be saved.

    Output:
        output_path (str or None): Full path to the saved difference TIFF file if successful,
                                   otherwise None if input files are not found.
    """
    if not os.path.exists(tif_before_path):
        print(f"[ERROR] File not found: {tif_before_path}")
        return None
    if not os.path.exists(tif_after_path):
        print(f"[ERROR] File not found: {tif_after_path}")
        return None

    date_before = os.path.splitext(os.path.basename(tif_before_path))[0]
    date_after = os.path.splitext(os.path.basename(tif_after_path))[0]

    os.makedirs(OUTPUT, exist_ok=True)
    output_filename = f"difference_{date_before}_{date_after}.tif"
    output_path = os.path.join(OUTPUT, output_filename)

    with rasterio.open(tif_after_path) as src1, rasterio.open(tif_before_path) as src2:
        img1 = src1.read(1)
        img2 = src2.read(1)
        meta = src1.meta.copy()

    diff = img1 - img2
    meta.update({"count": 1})

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(diff, 1)

    print(f"[INFO] Saved difference image to: {output_path}")
    return output_path

# =====================================================================

import pickle
import base64
import numpy as np

def decide_branch(ti, **kwargs):
    """
    Determines whether to continue processing with Earth Engine based on anomaly detection.

    Inputs:
        ti (TaskInstance): Airflow task instance used to pull XCom values.
        **kwargs: Additional keyword arguments passed by Airflow.

    Output:
        str: Task ID to follow next ('init_ee' if anomaly is detected, otherwise 'no_anomalies_path').
    """
    encoded = ti.xcom_pull(task_ids='detect_ndvi_anomalies', key='anomaly_labels')
    if encoded:
        anomaly_labels = pickle.loads(base64.b64decode(encoded))
        if np.any(anomaly_labels == -1):
            return 'init_ee'
    return 'no_anomalies_path'

# =====================================================================

def decide_early_path(**kwargs):
    """
    Chooses the processing path based on early anomaly detection results.

    Inputs:
        **kwargs: Should contain 'ti' (TaskInstance) used to pull XCom values.

    Output:
        str: Task ID to follow next ('download_historical_images' if anomaly is detected,
                                     otherwise 'early_no_anomalies_path').
    """
    ti = kwargs['ti']
    encoded = ti.xcom_pull(task_ids='detect_ndvi_anomalies', key='anomaly_labels')
    if encoded:
        anomaly_labels = pickle.loads(base64.b64decode(encoded))
        if np.any(anomaly_labels == -1):
            return 'download_historical_images'
    return 'early_no_anomalies_path'

# =====================================================================

def decide_path_after_composite(**kwargs):
    """
    Decides the path after running anomaly detection on the composite image.

    Inputs:
        **kwargs: Should contain 'ti' (TaskInstance) used to pull XCom values.

    Output:
        str: Task ID to follow next ('final_anomalies_path' if anomalies are detected,
                                     otherwise 'final_no_anomalies_path').
    """
    ti = kwargs['ti']
    encoded = ti.xcom_pull(task_ids='detect_ndvi_anomalies_on_composite', key='anomaly_labels')

    if encoded is None:
        return 'final_no_anomalies_path'

    anomaly_labels = pickle.loads(base64.b64decode(encoded))
    if np.any(anomaly_labels == -1):
        return 'final_anomalies_path'
    else:
        return 'final_no_anomalies_path'

################################################""
# def decide_early_path(**kwargs):
#     ti = kwargs['ti']
#     # Example logic: pull anomaly_labels from XCom
#     encoded = ti.xcom_pull(task_ids='detect_ndvi_anomalies', key='anomaly_labels')
#     if encoded:
#         anomaly_labels = pickle.loads(base64.b64decode(encoded))
#         # If any label is -1, it means anomaly found, go process EE branch
#         if np.any(anomaly_labels == -1):
#             return 'process_ee'
#     # Else no anomaly, follow early_no_anomalies_path
#     return 'early_no_anomalies_path'
#################################################"


def detect_ndvi_anomalies_with_voting(
    ndvi_before_stress_path,
    ndvi_after_stress_path,
    valid_mask_override=None,
    plot=True,
    ti=None
):
    import numpy as np
    import rasterio
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from scipy.ndimage import binary_opening, binary_closing
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.cm as cm
    import pickle
    import base64
    
    """
    Detects anomalies between two NDVI images using ensemble Isolation Forests.
    
    Inputs:
        ndvi_before_stress_path (str): Path to the pre-stress NDVI TIFF file.
        ndvi_after_stress_path (str): Path to the post-stress NDVI TIFF file.
        valid_mask_override (np.ndarray, optional): Boolean mask to specify valid pixels.
        plot (bool): If True, enables plotting (currently unused).
        ti (TaskInstance): Airflow task instance used to push results via XCom.

    Outputs:
        None. Pushes the following arrays (base64-encoded pickles) to XCom:
            - anomaly_labels
            - ndvi_diff
            - severity_score
            - combined_neg_mask
            - combined_pos_mask
            - other_mask
    """    

    EPSILON = 1e-8

    # Read NDVI before and after
    with rasterio.open(ndvi_before_stress_path) as src:
        ndvi_before_stress = src.read(1).astype(np.float32)
        if src.nodata is not None:
            ndvi_before_stress[ndvi_before_stress == src.nodata] = np.nan
    with rasterio.open(ndvi_after_stress_path) as src:
        ndvi_after_stress = src.read(1).astype(np.float32)
        if src.nodata is not None:
            ndvi_after_stress[ndvi_after_stress == src.nodata] = np.nan

    if ndvi_before_stress.shape != ndvi_after_stress.shape:
        raise ValueError("ndvi_before_stress and ndvi_after_stress must have the same shape.")

    ndvi_diff = ndvi_after_stress - ndvi_before_stress
    valid_mask = valid_mask_override if valid_mask_override is not None else \
        (~np.isnan(ndvi_before_stress) & ~np.isnan(ndvi_after_stress))
    severity_score = (ndvi_after_stress - ndvi_before_stress) / \
        (ndvi_before_stress + ndvi_after_stress + EPSILON)

    # Run ensemble Isolation Forest
    severity_flat = severity_score[valid_mask].reshape(-1, 1)
    diff_flat = ndvi_diff[valid_mask].reshape(-1, 1)
    n_models = 20
    majority_threshold = int(n_models * 0.5)

    votes_severity = np.zeros((n_models, severity_flat.shape[0]), dtype=int)
    votes_diff = np.zeros((n_models, diff_flat.shape[0]), dtype=int)

    for i in range(n_models):
        model_s = IsolationForest(n_estimators=100, random_state=i)
        votes_severity[i] = model_s.fit_predict(severity_flat)
        model_d = IsolationForest(n_estimators=100, random_state=i)
        votes_diff[i] = model_d.fit_predict(diff_flat)

    anomaly_votes_severity = np.sum(votes_severity == -1, axis=0)
    anomaly_votes_diff = np.sum(votes_diff == -1, axis=0)

    decision_severity = np.where(anomaly_votes_severity >= majority_threshold, -1, 1)
    decision_diff = np.where(anomaly_votes_diff >= majority_threshold, -1, 1)

    severity_labels = np.full(ndvi_diff.shape, 1)
    severity_labels[valid_mask] = decision_severity
    diff_labels = np.full(ndvi_diff.shape, 1)
    diff_labels[valid_mask] = decision_diff

    # Cleaning masks
    structure = np.ones((1, 1), dtype=bool)

    # Severity masks
    severity_anomaly_mask = (severity_labels == -1)
    severity_neg_mask = binary_closing(binary_opening(severity_anomaly_mask & (ndvi_diff < 0), structure=structure), structure=structure)
    severity_pos_mask = binary_closing(binary_opening(severity_anomaly_mask & (ndvi_diff > 0), structure=structure), structure=structure)
    severity_normal_mask = ~severity_anomaly_mask & valid_mask

    # Drop/swap negative severity mask
    neg_min = np.nanmin(severity_score[severity_neg_mask]) if np.any(severity_neg_mask) else np.nan
    normal_min = np.nanmin(severity_score[severity_normal_mask]) if np.any(severity_normal_mask) else np.nan
    if not np.isnan(neg_min) and not np.isnan(normal_min) and neg_min >= normal_min:
        tmp = severity_neg_mask.copy()
        severity_neg_mask = severity_normal_mask.copy()
        severity_normal_mask = tmp
    if np.sum(severity_neg_mask) > 0 and np.sum(severity_normal_mask) > 0:
        worst_normal = np.nanmin(severity_score[severity_normal_mask])
        weak_neg = severity_neg_mask & (severity_score > worst_normal)
        severity_neg_mask[weak_neg] = False
        severity_normal_mask[weak_neg] = True

    # Drop/swap positive severity mask
    pos_max = np.nanmax(severity_score[severity_pos_mask]) if np.any(severity_pos_mask) else np.nan
    normal_max = np.nanmax(severity_score[severity_normal_mask]) if np.any(severity_normal_mask) else np.nan
    if not np.isnan(pos_max) and not np.isnan(normal_max) and pos_max <= normal_max:
        tmp = severity_pos_mask.copy()
        severity_pos_mask = severity_normal_mask.copy()
        severity_normal_mask = tmp
    if np.sum(severity_pos_mask) > 0 and np.sum(severity_normal_mask) > 0:
        best_normal = np.nanmax(severity_score[severity_normal_mask])
        weak_pos = severity_pos_mask & (severity_score < best_normal)
        severity_pos_mask[weak_pos] = False
        severity_normal_mask[weak_pos] = True

    # Diff masks
    diff_anomaly_mask = (diff_labels == -1)
    diff_neg_mask = binary_closing(binary_opening(diff_anomaly_mask & (ndvi_diff < 0), structure=structure), structure=structure)
    diff_pos_mask = binary_closing(binary_opening(diff_anomaly_mask & (ndvi_diff > 0), structure=structure), structure=structure)
    diff_normal_mask = ~diff_anomaly_mask & valid_mask

    # Drop/swap negative diff mask
    neg_min = np.nanmin(ndvi_diff[diff_neg_mask]) if np.any(diff_neg_mask) else np.nan
    normal_min = np.nanmin(ndvi_diff[diff_normal_mask]) if np.any(diff_normal_mask) else np.nan
    if not np.isnan(neg_min) and not np.isnan(normal_min) and neg_min >= normal_min:
        tmp = diff_neg_mask.copy()
        diff_neg_mask = diff_normal_mask.copy()
        diff_normal_mask = tmp
    if np.sum(diff_neg_mask) > 0 and np.sum(diff_normal_mask) > 0:
        worst_normal = np.nanmin(ndvi_diff[diff_normal_mask])
        weak_neg = diff_neg_mask & (ndvi_diff > worst_normal)
        diff_neg_mask[weak_neg] = False
        diff_normal_mask[weak_neg] = True

    # Drop/swap positive diff mask
    pos_max = np.nanmax(ndvi_diff[diff_pos_mask]) if np.any(diff_pos_mask) else np.nan
    normal_max = np.nanmax(ndvi_diff[diff_normal_mask]) if np.any(diff_normal_mask) else np.nan
    if not np.isnan(pos_max) and not np.isnan(normal_max) and pos_max <= normal_max:
        tmp = diff_pos_mask.copy()
        diff_pos_mask = diff_normal_mask.copy()
        diff_normal_mask = tmp
    if np.sum(diff_pos_mask) > 0 and np.sum(diff_normal_mask) > 0:
        best_normal = np.nanmax(ndvi_diff[diff_normal_mask])
        weak_pos = diff_pos_mask & (ndvi_diff < best_normal)
        diff_pos_mask[weak_pos] = False
        diff_normal_mask[weak_pos] = True

    # Combine masks
    combined_neg_mask = severity_neg_mask | diff_neg_mask
    combined_pos_mask = severity_pos_mask | diff_pos_mask
    combined_anomaly_mask = combined_neg_mask | combined_pos_mask
    other_mask = ~combined_anomaly_mask & valid_mask

    anomaly_labels = np.full(ndvi_diff.shape, 1)
    anomaly_labels[combined_anomaly_mask] = -1



    # Push results to XCom (serialize + base64 encode)
    def serialize_and_encode(array):
        return base64.b64encode(pickle.dumps(array)).decode('utf-8')

    ti.xcom_push(key='anomaly_labels', value=serialize_and_encode(anomaly_labels))
    ti.xcom_push(key='ndvi_diff', value=serialize_and_encode(ndvi_diff))
    ti.xcom_push(key='severity_score', value=serialize_and_encode(severity_score))
    ti.xcom_push(key='combined_neg_mask', value=serialize_and_encode(combined_neg_mask))
    ti.xcom_push(key='combined_pos_mask', value=serialize_and_encode(combined_pos_mask))
    ti.xcom_push(key='other_mask', value=serialize_and_encode(other_mask))



 


def decide_after_sog_path(ti, **kwargs):
    """
    Decides the processing branch based on the difference between two SOG dates.
    
    Inputs:
        ti (TaskInstance): Used to pull SOG dates from XCom.
        **kwargs: Additional arguments passed from Airflow context.

    Output:
        str: Task ID to continue execution:
            - 'early_anomalies_path' if dates are within 7 days.
            - 'find_simulated_dates' if dates differ by more than 7 days.
    """
    sog_date_neg = ti.xcom_pull(task_ids='detect_SOG_date_both', key='sog_date_neg')
    sog_date_normal = ti.xcom_pull(task_ids='detect_SOG_date_both', key='sog_date_normal')

    # Convert to datetime (they are strings like '2025-03-10 00:00:00')
    sog_date_1 = datetime.strptime(sog_date_neg, "%Y-%m-%d %H:%M:%S")
    sog_date_2 = datetime.strptime(sog_date_normal, "%Y-%m-%d %H:%M:%S")

    # Compute difference
    difference = (sog_date_1 - sog_date_2).days  
    #difference = 13 # this is a test 

    print(f"Difference in days: {difference}")

    # Push difference to XCom if you want to use later
    ti.xcom_push(key='difference', value=difference)

    if abs(difference) < 7: 
        print("SOG dates are within 7 days. Continuing to 'anomalies_path'.")
        return 'early_anomalies_path'
    else:
        print("SOG dates differ by more than 7 days. Continuing to composite image '.")

        return 'find_simulated_dates'  


#############################################################


import os
import pickle
import base64
import numpy as np
import pandas as pd
import rasterio
def get_masked_mean(tif_path, mask_path, ti):
    """
    Computes mean NDVI values across time using different anomaly masks.

    Inputs:
        tif_path (str): Path to the target NDVI image (unused in this function).
        mask_path (str): Path to a mask file (unused).
        ti (TaskInstance): Used to pull combined_neg_mask and other_mask from XCom.

    Output:
        Tuple of pd.DataFrame:
            - df_neg: Time series of NDVI means over negative anomaly mask.
            - df_normal: Time series of NDVI means over normal pixels.
    """

    print(f"Loading masks from XCom...")

    # Pull masks from XCom
    encoded_neg_mask = ti.xcom_pull(task_ids='detect_ndvi_anomalies', key='combined_neg_mask')
    encoded_other_mask = ti.xcom_pull(task_ids='detect_ndvi_anomalies', key='other_mask')

    # Decode
    combined_neg_mask = pickle.loads(base64.b64decode(encoded_neg_mask))
    other_mask = pickle.loads(base64.b64decode(encoded_other_mask))

    # Find historical_TIFs folder
    hist_folder = '/home/user/airflow/historical_TIFs'
    tif_files = sorted([f for f in os.listdir(hist_folder) if f.endswith('.tif')])

    print(f"Found {len(tif_files)} historical NDVI images.")

    # Prepare lists
    dates = []
    neg_means = []
    normal_means = []

    for tif in tif_files:
        date = tif.split('_')[0]
        tif_path = os.path.join(hist_folder, tif)

        with rasterio.open(tif_path) as src:
            ndvi = src.read(1).astype(np.float32)
            if src.nodata is not None:
                ndvi[ndvi == src.nodata] = np.nan

        neg_mean = np.nanmean(ndvi[combined_neg_mask]) if np.any(combined_neg_mask) else np.nan
        normal_mean = np.nanmean(ndvi[other_mask]) if np.any(other_mask) else np.nan

        dates.append(date)
        neg_means.append(neg_mean)
        normal_means.append(normal_mean)

        print(f"Date: {date} | Neg mean: {neg_mean:.4f} | Normal mean: {normal_mean:.4f}")

    # Build two separate DataFrames
    df_neg = pd.DataFrame({
        'date': dates,
        'y': neg_means
    })

    df_normal = pd.DataFrame({
        'date': dates,
        'y': normal_means
    })

    print("\n=== Time series over negative anomalies mask ===")
    print(df_neg)

    print("\n=== Time series over normal mask ===")
    print(df_normal)

    return df_neg, df_normal


#############################################################





import pandas as pd
import ruptures as rpt

def detect_SOG_date(df_ndvi, model="rbf"):
    """
    Detects the Start of Growth (SOG) date in a time series using change point detection.

    Inputs:
        df_ndvi (pd.DataFrame): DataFrame with 'date' and 'y' columns representing NDVI values.
        model (str): Model type for change point detection (default is 'rbf').

    Output:
        pd.Timestamp: The date corresponding to the first detected change point in the NDVI signal.
    """

    # Ensure datetime and sort
    df_ndvi = df_ndvi.copy()
    df_ndvi['date'] = pd.to_datetime(df_ndvi['date'])
    df_ndvi = df_ndvi.sort_values('date')

    # Extract values
    ndvi_signal = df_ndvi['y'].values   
    # Change point detection
    ndvi_algo = rpt.Binseg(model=model).fit(ndvi_signal)
    ndvi_change_points = ndvi_algo.predict(n_bkps=2)
    first_ndvi_cp = ndvi_change_points[0]

    ndvi_SOG_date = df_ndvi['date'].iloc[first_ndvi_cp]

    print(f"NDVI first change point date: {ndvi_SOG_date}")

    # ✅ Return full pandas Timestamp objects
    return ndvi_SOG_date



def detect_sog_for_both_masks(ti, **kwargs):
    """
    Detects and compares SOG dates for both the negative anomaly mask and the normal region.

    Inputs:
        ti (TaskInstance): Used to pull the two time series DataFrames from XCom (from task 'compute_mean').
        **kwargs: Additional context from Airflow (not used here).

    Outputs:
        tuple(str, str): Two SOG dates in 'YYYY-MM-DD' format for negative and normal masks.
                         Also pushes both to XCom with keys 'sog_date_neg' and 'sog_date_normal'.
    """

    # Option 1: if the previous task (e.g. get_masked_mean) returned the two DataFrames directly
    df_neg, df_normal = ti.xcom_pull(task_ids='compute_mean')

    sog_date_neg = detect_SOG_date(df_neg)
    sog_date_normal = detect_SOG_date(df_normal)

    print(f"\nSOG date for negative anomalies mask: {sog_date_neg}")
    print(f"SOG date for normal mask: {sog_date_normal}")

    # Optionally push to XCom if you need later
    ti.xcom_push(key='sog_date_neg', value=str(sog_date_neg))
    ti.xcom_push(key='sog_date_normal', value=str(sog_date_normal))
    print("SOG dates pushed to XCom.", sog_date_neg, sog_date_normal)
    # Return both if useful
    return  (
    sog_date_neg.strftime("%Y-%m-%d"),
    sog_date_normal.strftime("%Y-%m-%d")
)
#######################""  
def find_simulated_dates_wrapper(date_after_path, **kwargs):
    """
    Wrapper function that computes the date_before_path by shifting date_after_path based on SOG difference.

    Inputs:
        date_after_path (str): Path to the NDVI image taken after the event.
        **kwargs: Must include 'ti' (TaskInstance) to pull the 'difference' from XCom.

    Output:
        dict: Result from calling `find_simulated_dates()` containing simulated before/after dates.
    """
    ti = kwargs['ti']
    
    print(f"Using date_after_path: {date_after_path}")

    # Pull the difference pushed by branching_2
    sog_shift = ti.xcom_pull(task_ids='branching_2', key='difference')
    print(f"Pulled difference (sog_shift) from branching_2: {sog_shift}")

    if sog_shift is None:
        raise ValueError("difference XCom not found in branching_2")

    sog_shift = int(sog_shift)
    print(f"Using sog_shift as int: {sog_shift}")

    # Extract date from given date_after_path (e.g., '2025-01-24_NDVI.tif' -> datetime)
    date_after = extract_date_from_filename(date_after_path)
    print(f"Extracted date_after: {date_after}")

    # Calculate date_before by subtracting sog_shift days
    date_before = date_after - timedelta(days=sog_shift)
    print(f"Calculated date_before: {date_before}")

    # Build date_before_path assuming same folder and filename pattern
    folder = os.path.dirname(date_after_path)
    date_before_filename = f"{date_before.strftime('%Y-%m-%d')}_NDVI.tif"
    date_before_path = os.path.join(folder, date_before_filename)
    print(f"Computed date_before_path: {date_before_path}")

    # Call the actual function
    return find_simulated_dates(date_before_path, date_after_path, **kwargs)


##################################################
from datetime import datetime
import pandas as pd
def find_simulated_dates(date_before, date_after, **kwargs):
    """
    Computes simulated NDVI date range by subtracting the SOG difference from both before and after dates.

    Inputs:
        date_before (str): File path to the NDVI image before the event.
        date_after (str): File path to the NDVI image after the event.
        **kwargs: Must include 'ti' (TaskInstance) to pull 'difference' from XCom.

    Output:
        dict: Contains:
            - 'simulated_date_before' (str): Adjusted before date (YYYY-MM-DD)
            - 'simulated_date_after' (str): Adjusted after date (YYYY-MM-DD)
    """
    ti = kwargs['ti']
    print("testing find_simulated_dates")

    # Pull 'difference' from branching_2 (consistent with your push)
    sog_shift = ti.xcom_pull(task_ids='branching_2', key='difference')
    print("All XCom from branching_2:", ti.xcom_pull(task_ids='branching_2'))
    print("Pulled sog_shift (difference):", sog_shift)

    if sog_shift is None:
        raise ValueError("difference not found in XCom from branching_2")

    sog_shift = int(sog_shift)  # ensure integer

    before_date = extract_date_from_filename(date_before)
    after_date = extract_date_from_filename(date_after)

    simulated_date_before = before_date - timedelta(days=sog_shift)
    simulated_date_after = after_date - timedelta(days=sog_shift)

    print(f"Simulated date before: {simulated_date_before}")
    print(f"Simulated date after: {simulated_date_after}")

    return {
        'simulated_date_before': simulated_date_before.strftime('%Y-%m-%d'),
        'simulated_date_after': simulated_date_after.strftime('%Y-%m-%d')
    }


################################################"
import ee
import os
import requests
def download_historical_images(start_date, end_date, roi_coords):
    """
    Downloads Sentinel-2 NDVI images from Google Earth Engine for the given date range and ROI.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        roi_coords (list): Polygon coordinates for the region of interest as [[lon, lat], ...].
        
    Saves:
        GeoTIFF NDVI images in a local folder named 'historical_TIFs'.
    """

    try:
        credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, SERVICE_ACCOUNT_KEY_PATH)
        ee.Initialize(credentials)    
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        return

    roi = ee.Geometry.Polygon([roi_coords])

    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))

    image_list = collection.toList(collection.size())
    n_images = image_list.size().getInfo()
    print(f"Found {n_images} images between {start_date} and {end_date}")

    # Create historical_TIF folder if it doesn't exist
    hist_folder = os.path.join(os.getcwd(), "historical_TIFs")
    os.makedirs(hist_folder, exist_ok=True)
    for i in range(n_images):
        image = ee.Image(image_list.get(i))

        ndvi = image.normalizedDifference(['B8', 'B4']).clip(roi)

        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()

        url = ndvi.getDownloadURL({
            'scale': 10,
            'crs': 'EPSG:4326',
            'region': roi.toGeoJSONString(),
            'format': 'GeoTIFF'
        })

        print(f"Downloading NDVI for {date}")

        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join(hist_folder, f"{date}_NDVI.tif")
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Saved NDVI image to {file_path}")
        else:
            print(f"Failed to download image for {date}: Status code {response.status_code}")
    print("Finished downloading historical images.")




#################################################


    
###############################################################################
import os
import re
from datetime import datetime, timedelta

def extract_date_from_filename(tif_path):
    """
    Extracts a date string from a filename and returns it as a datetime object.

    Args:
        tif_path (str): Filename or path containing a date in 'YYYY-MM-DD' format.
    
    Returns:
        datetime: Extracted date as a datetime object.
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2})", tif_path)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d")
    else:
        raise ValueError(f"No date found in filename: {tif_path}")
    
    
    
#################################################################################"



def get_closest_available_date(target_date, tif_list):
    """
    Finds the closest date-matching TIF file to a given target date.

    Args:
        target_date (datetime): Date to match.
        tif_list (list): List of .tif file paths.
    
    Returns:
        str: Path to the closest-matching .tif file.
    """
    closest_tif = None
    min_diff = float('inf')
    for tif in tif_list:
        tif_date = extract_date_from_filename(tif)
        diff = abs((tif_date - target_date).days)
        if diff < min_diff:
            min_diff = diff
            closest_tif = tif
    if closest_tif is None:
        raise ValueError("No suitable NDVI TIF found.")
    return closest_tif

#################################################################################""""
from datetime import datetime


def find_closest_available_dates(ndvi_tif_paths, **kwargs):
    """
    Finds the closest NDVI TIFs to simulated before/after dates (pulled from XCom).

    Args:
        ndvi_tif_paths (list): List of available NDVI .tif paths.
        **kwargs: Contains task instance (`ti`) to pull simulated dates from XCom.
    
    Returns:
        dict: Paths and original dates of the closest before/after TIFs.
    """
    ti = kwargs['ti']

    # Pull simulated dates
    simulated_dates = ti.xcom_pull(task_ids='find_simulated_dates')
    if not simulated_dates:
        raise ValueError("Simulated dates not found in XCom")

    from datetime import datetime

    simulated_before = datetime.strptime(simulated_dates['simulated_date_before'], '%Y-%m-%d')
    simulated_after = datetime.strptime(simulated_dates['simulated_date_after'], '%Y-%m-%d')

    def find_closest(tif_list, target_date):
        closest_tif = None
        closest_date = None
        min_diff = float('inf')
        for tif_path in tif_list:
            tif_date = extract_date_from_filename(tif_path)
            diff = abs((tif_date - target_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_tif = tif_path
                closest_date = tif_date
        return closest_tif, closest_date

    closest_before, closest_before_date = find_closest(ndvi_tif_paths, simulated_before)
    closest_after, closest_after_date = find_closest(ndvi_tif_paths, simulated_after)

    if not closest_before or not closest_after:
        raise ValueError("Could not find closest TIFs to simulated dates")

    print(f"Closest TIF before simulated date: {closest_before}")
    print(f"Closest TIF after simulated date: {closest_after}")

    # Return XCom dict
    return {
        'simulated_before_tif': closest_before,     # full path
        'simulated_after_tif': closest_after,       # full path
        'original_before_date': closest_before_date.strftime('%Y-%m-%d'),
        'original_after_date': closest_after_date.strftime('%Y-%m-%d')
    }

#################################################################################""""
import numpy as np
import rasterio

def create_composite_image(image1_path, image2_path, mask_array, output_path, **kwargs):
    """
    Creates a composite image by blending two images based on a binary mask.

    Args:
        image1_path (str): Path to first image (used where mask is True).
        image2_path (str): Path to second image (used where mask is False).
        mask_array (np.ndarray or str): Binary mask array or path to .npy file.
        output_path (str): File path to save the composite image.
    
    Saves:
        Composite image as GeoTIFF at output_path.
    """

    # If mask_array is a string (filepath), load it
    if isinstance(mask_array, str):
        mask_array = np.load(mask_array)

    with rasterio.open(image1_path) as src1, rasterio.open(image2_path) as src2:
        if (src1.width != src2.width or src1.height != src2.height):
            raise ValueError("Images have different dimensions")
        if src1.count != src2.count:
            raise ValueError("Images have different band counts")

        img1 = src1.read()
        img2 = src2.read()

        nodata1 = src1.nodata
        nodata2 = src2.nodata

        mask = mask_array.astype(bool)
        
        if mask.shape != (src1.height, src1.width):
            raise ValueError("Mask shape mismatch with images")

        if nodata1 is not None:
            nodata_mask1 = np.any(img1 == nodata1, axis=0)
        else:
            nodata_mask1 = np.zeros((src1.height, src1.width), dtype=bool)

        if nodata2 is not None:
            nodata_mask2 = np.any(img2 == nodata2, axis=0)
        else:
            nodata_mask2 = np.zeros((src2.height, src2.width), dtype=bool)

        composite = np.empty_like(img1)

        for b in range(img1.shape[0]):
            composite[b] = np.where(mask, img1[b], img2[b])
            composite[b][nodata_mask1 & nodata_mask2] = nodata1 if nodata1 is not None else 0

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=src1.height,
            width=src1.width,
            count=src1.count,
            dtype=img1.dtype,
            crs=src1.crs,
            transform=src1.transform,
            nodata=nodata1 if nodata1 is not None else None
        ) as dst:
            dst.write(composite)

    print(f"Composite image created at: {output_path}")
#################################################################################"""
import base64
import pickle

def deserialize_and_decode(encoded_str):
    """
    Decodes and deserializes a base64-encoded string (typically for masks).

    Args:
        encoded_str (str): Base64-encoded, pickled Python object.
    
    Returns:
        object: Decoded and deserialized Python object.
    """
    decoded = base64.b64decode(encoded_str)
    obj = pickle.loads(decoded)
    return obj

#################################################################################""""
import os

import os

def create_composite_before_wrapper(**kwargs):
    """
    Wrapper that creates a composite NDVI image for the 'before' case using:
    - Simulated TIF path (from XCom)
    - Original historical TIF (based on closest date)
    - Mask (decoded from XCom)
    
    Returns:
        None or result from create_composite_image (writes to file).
    """
    ti = kwargs['ti']

    # Pull the whole dict returned by find_closest_tifs from XCom
    closest_tifs = ti.xcom_pull(task_ids='find_closest_tifs', key='return_value')
    if not closest_tifs:
        raise ValueError("No data found in XCom from find_closest_tifs")

    simulated_before_tif = closest_tifs.get('simulated_before_tif')
    date_str = closest_tifs.get('original_before_date')

    print("Date string from XCom for before composite:", date_str)

    if not simulated_before_tif or not date_str:
        raise ValueError("Expected keys missing in XCom dict for before composite")

    # Construct original_before_tif path dynamically based on the date_str
    base_folder = '/home/user/airflow/historical_TIFs'
    original_before_tif = os.path.join(base_folder, f"{date_str}_NDVI.tif")
    
    # Pull serialized mask string from XCom and deserialize
    encoded_mask = ti.xcom_pull(task_ids='detect_ndvi_anomalies', key='other_mask')
    if encoded_mask is None:
        raise ValueError("No mask found in XCom under 'other_mask'")
    mask_array = deserialize_and_decode(encoded_mask)

    output_dir = '/home/user/airflow/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'composite_before.tif')

    return create_composite_image(
        image1_path=simulated_before_tif,
        image2_path=original_before_tif,
        mask_array=mask_array,
        output_path=output_path
    )


def create_composite_after_wrapper(**kwargs):
    """
    Wrapper that creates a composite NDVI image for the 'after' case using:
    - Simulated TIF path (from XCom)
    - Original historical TIF (based on closest date)
    - Mask (decoded from XCom)
    
    Returns:
        None or result from create_composite_image (writes to file).
    """
    ti = kwargs['ti']
    print("Creating composite after wrapper...")
    # Pull the whole dict returned by find_closest_tifs from XCom
    closest_tifs = ti.xcom_pull(task_ids='find_closest_tifs', key='return_value')
    if not closest_tifs:
        raise ValueError("No data found in XCom from find_closest_tifs")

    simulated_after_tif = closest_tifs.get('simulated_after_tif')
    date_str = closest_tifs.get('original_after_date')

    print("Date string from XCom for after composite:", date_str)

    if not simulated_after_tif or not date_str:
        raise ValueError("Expected keys missing in XCom dict for after composite")

    # Construct original_after_tif path dynamically based on the date_str
    base_folder = '/home/user/airflow/historical_TIFs'
    original_after_tif = os.path.join(base_folder, f"{date_str}_NDVI.tif")

    # Pull serialized mask string from XCom and deserialize
    encoded_mask = ti.xcom_pull(task_ids='detect_ndvi_anomalies', key='other_mask')
    if encoded_mask is None:
        raise ValueError("No mask found in XCom under 'other_mask'")
    mask_array = deserialize_and_decode(encoded_mask)
    print("Mask array shape:", mask_array.shape)
    output_dir = '/home/user/airflow/output'
    if not os.path.exists(output_dir):
 
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'composite_after.tif')


    return create_composite_image(
        image1_path=simulated_after_tif,
        image2_path=original_after_tif,
        mask_array=mask_array,
        output_path=output_path
    )

#################################################################################"""
#################################################################################""""

import os
def find_closest_tifs_wrapper(**kwargs):
    """
    Wrapper that reads all historical TIFs in the default folder and finds
    the closest matching files to simulated before/after dates.
    
    Returns:
        dict: Result from `find_closest_available_dates()`.
    """
    base_folder = '/home/user/airflow/historical_TIFs'
    ndvi_tif_paths = [
        os.path.join(base_folder, tif)
        for tif in os.listdir(base_folder)
        if tif.endswith('.tif')
    ]
    return find_closest_available_dates(ndvi_tif_paths, **kwargs)
