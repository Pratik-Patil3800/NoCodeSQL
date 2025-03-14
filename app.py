import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sqlalchemy as sa
from sqlalchemy import inspect, MetaData, create_engine
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and tokenizer
logger.info("Loading Natural-SQL-7B model...")

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained("chatdb/natural-sql-7b")

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device.upper()}")

model = AutoModelForCausalLM.from_pretrained(
    "chatdb/natural-sql-7b",
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")

def get_table_ddl(engine):
    """Extracts database schema as CREATE TABLE statements"""
    inspector = inspect(engine)
    metadata = MetaData()
    metadata.reflect(bind=engine)

    ddl_statements = []

    for table_name in inspector.get_table_names():
        columns = []
        for column in inspector.get_columns(table_name):
            col_type = str(column['type'])
            nullable = "NULL" if column.get('nullable', True) else "NOT NULL"
            primary_key = " PRIMARY KEY" if column.get('primary_key', False) else ""
            columns.append(f"    {column['name']} {col_type} {nullable}{primary_key}")

        foreign_keys = []
        for fk in inspector.get_foreign_keys(table_name):
            src_cols = ", ".join(fk['constrained_columns'])
            ref_cols = ", ".join(fk['referred_columns'])
            foreign_keys.append(f"    FOREIGN KEY ({src_cols}) REFERENCES {fk['referred_table']}({ref_cols})")

        create_stmt = f"CREATE TABLE {table_name} (\n"
        create_stmt += ",\n".join(columns)

        if foreign_keys:
            create_stmt += ",\n" + ",\n".join(foreign_keys)

        create_stmt += "\n);"
        ddl_statements.append(create_stmt)

    return "\n\n".join(ddl_statements)

def generate_sql(question, schema_ddl):
    """Generates SQL query from English input"""
    prompt = f"""# Task
Generate a SQL query to answer the following question: `{question}`

### PostgreSQL Database Schema
The query will run on a database with the following schema:

{schema_ddl}

# SQL
Here is the SQL query that answers the question: `{question}`
```sql
"""

    logger.info(f"Generating SQL for: {question}")

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate SQL
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=100001,
        pad_token_id=100001,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
    )

    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    sql_query = output.split("```sql")[-1].strip().rstrip('`').strip()

    return sql_query

# === DATABASE CONNECTION ===
# Example: SQLite (Change to PostgreSQL/MySQL as needed)
DATABASE_URI = "mysql://root:LhGOcVtnNzyvxBBrOCZZdLmLmSQVcBUi@shinkansen.proxy.rlwy.net:53689/railway"  # Update this for your database
engine = create_engine(DATABASE_URI)

logger.info("Extracting database schema...")
schema_ddl = get_table_ddl(engine)
logger.info("Schema extraction complete.")

# === RUN EXAMPLE ===
while True:
    question = input("enter query")
    if question=="exit" :
      break
    sql_query = generate_sql(question, schema_ddl)
    print("\nGenerated SQL Query:\n", sql_query)
