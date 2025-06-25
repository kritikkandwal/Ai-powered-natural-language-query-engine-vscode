import os
import re
import sqlparse
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Load fine-tuned model if exists, else fallback
model_dir = os.path.join(os.path.dirname(__file__), "../../models/fine_tuned_model")

try:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
except:
    # fallback to pretrained if not fine-tuned yet
    model_name = "mrm8488/t5-base-finetuned-wikiSQL"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def generate_sql(natural_language):
    input_text = f"translate English to SQL: {natural_language}"
    result = generator(
        input_text,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    sql_raw = result[0]['generated_text']
    cleaned_sql = clean_sql(sql_raw, natural_language)  # ✅ pass both args
    return cleaned_sql

def clean_sql(sql, user_input):
    import re
    sql = sql.replace('"', '').replace('`', '').strip()

    # Replace hallucinated words
    sql = sql.replace("table", "Employee")
    sql = sql.replace("Earnings ( $ )", "Salary")
    sql = re.sub(r'\bemployee\b', 'Employee', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bemp_id\b', 'EmpID', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bsalaries?\b', 'Salary', sql, flags=re.IGNORECASE)

    if "between" in user_input.lower():
        import re
        range_match = re.search(r'between\s+(\d+)\s+(and|-)\s+(\d+)', user_input.lower())
        if range_match:
            low, high = range_match.group(1), range_match.group(3)
            sql = re.sub(r'Salary.*?(\d+)', f'Salary BETWEEN {low} AND {high}', sql, flags=re.IGNORECASE)

    # Handle missing operator — context aware fallback
    match = re.search(r'Salary\s+(?![<>=])\s*(\d+)', sql)
    if match:
        number = match.group(1)
        if "less than" in user_input.lower():
            operator = "<"
        elif "greater than" in user_input.lower() or "more than" in user_input.lower() or "above" in user_input.lower():
            operator = ">"
        elif "equal to" in user_input.lower() or "equals" in user_input.lower():
            operator = "="
        else:
            operator = "="  # default if unsure

        sql = re.sub(r'Salary\s+(\d+)', f'Salary {operator} {number}', sql)

    # Add SELECT fallback
    if "SELECT" in sql.upper() and "FROM" in sql.upper():
        if not re.search(r'EmpID', sql, re.IGNORECASE) or not re.search(r'Salary', sql, re.IGNORECASE):
            sql = re.sub(r'SELECT\s+.*?\s+FROM', 'SELECT EmpID, Salary FROM', sql, flags=re.IGNORECASE)

    # Validate it’s a SELECT query
    if not is_valid_select(sql):
        raise ValueError("Only SELECT queries are allowed.")

    return sql




def is_valid_select(sql):
    parsed = sqlparse.parse(sql)
    for stmt in parsed:
        if stmt.get_type() != "SELECT":
            return False
        for token in stmt.flatten():
            if token.value.upper() in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", ";", "--", "TRUNCATE"]:
                return False
    return True
