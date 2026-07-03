import os
import pandas as pd
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine

def get_engine():
    load_dotenv()
    pw = quote_plus(os.environ["PG_PASSWORD"])
    return create_engine(f"postgresql+psycopg2://postgres:{pw}@localhost:5432/mind")

def query(sql, engine):
    return pd.read_sql(sql, engine)