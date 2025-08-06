from airflow import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.expanduser('~/airflow'))

from ndvi_utils import (
    compute_difference_tif,
    detect_ndvi_anomalies_with_voting,
    download_historical_images,
    get_masked_mean,
    decide_branch,
    find_simulated_dates_wrapper,
    init_ee,
    detect_sog_for_both_masks,
    create_composite_before_wrapper,
    find_simulated_dates,
    find_closest_tifs_wrapper,
    decide_path_after_composite,
    
    find_closest_available_dates,
    decide_after_sog_path,
    decide_early_path, 
    create_composite_after_wrapper
)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


with DAG(
    dag_id='ndvi_change_detection_pipeline_branching',
    default_args=default_args,
    description='NDVI pipeline with branching after anomaly detection',
    schedule=timedelta(days=1),
    start_date=datetime(2025, 7, 23),
    catchup=False,
    tags=['ndvi', 'change-detection'],
) as dag:

    compute_diff = PythonOperator(
        task_id='compute_difference_tif',
        python_callable=compute_difference_tif,
        op_kwargs={
            'tif_before_path': '/home/user/airflow/2025-01-09_NDVI.tif',
            'tif_after_path': '/home/user/airflow/2025-01-24_NDVI.tif',
            'OUTPUT': '/home/user/airflow/diff_folder',
        }
    )

    detect_anomalies = PythonOperator(
        task_id='detect_ndvi_anomalies',
        python_callable=detect_ndvi_anomalies_with_voting,
        op_kwargs={
            'ndvi_before_stress_path': '/home/user/airflow/2025-01-09_NDVI.tif',
            'ndvi_after_stress_path': '/home/user/airflow/2025-01-24_NDVI.tif',
        }
    )



    init_ee_task = PythonOperator(
        task_id='init_ee',
        python_callable=init_ee
    )

    start_of_year = datetime(datetime.now().year, 1, 1).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    download_historical_images_task = PythonOperator(
        task_id='download_historical_images',
        python_callable=download_historical_images,
        op_kwargs={
            'start_date': start_of_year,
            'end_date': end_date,
            'roi_coords': [
                [10.3036, 34.7687], [10.3018, 34.7676], [10.3029, 34.7664],
                [10.3044, 34.7647], [10.3048, 34.7643], [10.3068, 34.7620],
                [10.3070, 34.7620], [10.3073, 34.7622], [10.3074, 34.7621],
                [10.3089, 34.7630], [10.3080, 34.7640], [10.3074, 34.7646],
                [10.3054, 34.7668],
            ],
        }
    )

    compute_mean = PythonOperator(
        task_id='compute_mean',
        python_callable=get_masked_mean,
        op_kwargs={
            'tif_path': '/home/user/airflow/historical_folder/difference_2025-01-09_NDVI_2025-01-24_NDVI.tif',
            'mask_path': '/home/user/airflow/mask.tif',
        }
    )

    detect_sog = PythonOperator(
        task_id='detect_SOG_date_both',
        python_callable=detect_sog_for_both_masks,
    )



    find_simulated_dates_task = PythonOperator(
    task_id='find_simulated_dates',
    python_callable=find_simulated_dates_wrapper,
    op_kwargs={'date_after_path': '/home/user/airflow/historical_TIFs/2025-01-24_NDVI.tif'},
)


    find_closest_tifs_task = PythonOperator(
        task_id='find_closest_tifs',
        python_callable=find_closest_tifs_wrapper,
    )

    create_composite_before_task = PythonOperator(
        task_id='create_composite_before',
        python_callable=create_composite_before_wrapper,
    )


    create_composite_after_task = PythonOperator(
        task_id='create_composite_after',
        python_callable=create_composite_after_wrapper,
    )

    detect_anomalies_on_composite = PythonOperator(
        task_id='detect_ndvi_anomalies_on_composite',
        python_callable=detect_ndvi_anomalies_with_voting,
        op_kwargs={
            'ndvi_before_stress_path': '/home/user/airflow/output/composite_before.tif',
            'ndvi_after_stress_path': '/home/user/airflow/output/composite_after.tif',
        }
    )
    # Tasks
early_no_anomalies_path = EmptyOperator(task_id='early_no_anomalies_path')
process_ee = EmptyOperator(task_id='process_ee')

branching_early = BranchPythonOperator(
    task_id='branching_early',
    python_callable=decide_early_path,
)

branching_2 = BranchPythonOperator(
    task_id='branching_2',
    python_callable=decide_after_sog_path,
)

branching_3 = BranchPythonOperator(
    task_id='branching_3',
    python_callable=decide_path_after_composite,
)

final_no_anomalies_path = EmptyOperator(task_id='final_no_anomalies_path')
final_anomalies_path = EmptyOperator(task_id='final_anomalies_path')

early_anomalies_path = EmptyOperator(task_id='early_anomalies_path')

# Define dependencies:
init_ee_task >> compute_diff >> detect_anomalies >> branching_early

branching_early >> early_no_anomalies_path
branching_early >> download_historical_images_task

download_historical_images_task >> compute_mean >> detect_sog >> branching_2

branching_2 >> find_simulated_dates_task >> find_closest_tifs_task
branching_2 >> early_anomalies_path

find_closest_tifs_task >> create_composite_before_task >> create_composite_after_task >> detect_anomalies_on_composite >> branching_3

branching_3 >> final_no_anomalies_path
branching_3 >> final_anomalies_path
