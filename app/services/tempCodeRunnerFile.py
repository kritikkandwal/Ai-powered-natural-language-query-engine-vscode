from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import os
import re
import sqlparse
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Determine model path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, '../../models/fine_tuned_model')
PRETRAINED_MODEL = "mrm8488/t5-base-finetuned-wikiSQL"

# Define database schema
DATABASE_SCHEMA = "Employee(EmpID, Salary)"

try:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    logger.info("Loaded fine-tuned model successfully")
except Exception as e:
    logger.warning(f"Failed to load fine-tuned model: {str(e)}. Using pretrained model.")
    model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

# Create generator pipeline
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer
)

def generate_sql(natural_language):
    """Generate SQL from natural language input with schema context"""
    # Prepare prompt with schema
    input_text = f"Schema: {DATABASE_SCHEMA}. Question: {natural_language}"
    
    # Generate SQL
    result = generator(
        input_text,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    
    generated_sql = clean_sql(result[0]['generated_text'])
    return generated_sql

def clean_sql(sql):
    """Sanitize and format generated SQL"""
    # Remove unsafe characters
    sql = sql.replace('"', '').replace('`', '').strip()
    
    # Replace model generated names with our schema
    sql = re.sub(r'\bemployee\b', 'Employee', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bemp_id\b', 'EmpID', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bsalaries?\b', 'Salary', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\btable\b', 'Employee', sql, flags=re.IGNORECASE)
    
    # Ensure SELECT includes required columns
    if "SELECT" in sql.upper() and "FROM" in sql.upper():
        if not re.search(r'EmpID', sql, re.IGNORECASE) or not re.search(r'Salary', sql, re.IGNORECASE):
            sql = re.sub(
                r'SELECT\s+.*?\s+FROM',
                'SELECT EmpID, Salary FROM',
                sql,
                flags=re.IGNORECASE
            )
    
    # Validate query type
    if not is_valid_select(sql):
        raise ValueError("Only SELECT queries are allowed")
    
    return sql

def is_valid_select(sql):
    """Ensure only SELECT queries are allowed"""
    parsed = sqlparse.parse(sql)
    for stmt in parsed:
        if not stmt.get_type() == "SELECT":
            return False
        for token in stmt.flatten():
            if token.value.upper() in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", ";", "--"]:
                return False
    return True