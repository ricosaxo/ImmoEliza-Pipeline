from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta


# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'run_eliza_scripts',
    default_args=default_args,
    description='runs scraper, preprocessing, model scripts',
    schedule_interval='@weekly',
    start_date=datetime(2024, 12, 22),
    catchup=False,
)


run_scraper_script = BashOperator(
    task_id='run_scraper',
    bash_command='python3 scraper/scraper.py',
    dag=dag,
)

run_preprocessing_script = BashOperator(
    task_id='run_preprocessing',
    bash_command='python3 Preprocessing/preprocessing.py',  
    dag=dag,
)

run_model_script = BashOperator(
    task_id='run_model_script',
    bash_command='python3 model/project.py',  
    dag=dag,
)

git_push = BashOperator(
    task_id='git_push',
    bash_command='cd /Documents/GitHub/ImmoEliza-Pipeline && git add . && git commit -m "Auto commit" && git push origin main',
    dag=dag,
)

run_scraper_script >> run_preprocessing_script >> run_model_script >> git_push
