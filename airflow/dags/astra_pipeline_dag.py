from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.bash import BashOperator

# Default arguments for the DAG
default_args = {
    'owner': 'astra_admin',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'astra_forecast_pipeline',
    default_args=default_args,
    description='End-to-End ASTRA pipeline for NWP blending',
    schedule_interval=timedelta(hours=6),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['astra', 'forecast', 'blending'],
) as dag:

    # Task to run the pipeline
    # The pipeline script is located in /app/src/run_pipeline.py inside the container
    run_blending_pipeline = BashOperator(
        task_id='run_blending_pipeline',
        bash_command='cd /app && python src/run_pipeline.py',
        env={
            **os.environ,
            'PYTHONPATH': '/app/src',
            'PYTHONIOENCODING': 'utf-8'
        }
    )
    run_blending_pipeline