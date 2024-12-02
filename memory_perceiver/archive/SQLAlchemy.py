from sqlalchemy import create_engine
engine = create_engine('sqlite:///my_database.db')
table_name = "test"
with engine.connect() as connection:
    df.to_sql(table_name, con=connection, if_exists='append', index=False)

from sqlalchemy import create_engine
engine = create_engine('sqlite:///my_database.db')
table_name = "test0"
with engine.connect() as connection:
    df[['h_i', 'p_j']].to_sql(table_name, con=connection, if_exists='append', index=False)

query = """
SELECT test.*
FROM test
JOIN test0 ON test.h_i = test0.h_i;
"""

df = pd.read_sql(query, engine)