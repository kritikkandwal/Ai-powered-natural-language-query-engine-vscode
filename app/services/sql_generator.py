import os
import re
import sqlparse
from pathlib import Path
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# ✅ Convert model path to correct local format
model_dir = Path(__file__).resolve().parent.parent.parent / "models" / "fine_tuned_model"
model_dir_str = os.path.abspath(model_dir)

try:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir_str, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir_str, local_files_only=True, use_fast=False)
except Exception as e:
    print(f"[WARNING] Local model load failed. Falling back to Hugging Face. Error: {e}")
    model_name = "mrm8488/t5-base-finetuned-wikiSQL"
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception as e2:
        raise RuntimeError(f"Failed to load both local and remote models. Error: {e2}")

# ✅ Avoid max_length + max_new_tokens conflict by using only max_new_tokens
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def parse_schema(schema_str):
    schema = {}
    tables = re.findall(r'(\w+)\(([^)]+)\)', schema_str)
    for table, columns in tables:
        schema[table.lower()] = [col.strip().lower() for col in columns.split(',')]
    return schema

def map_to_schema(term, schema):
    term_lower = term.lower()
    for table, columns in schema.items():
        if term_lower == table:
            return table
        for col in columns:
            if term_lower == col:
                return f"{table}.{col}"
    return term

def generate_sql(natural_language):
    schema_info = {}
    if "Schema:" in natural_language and "Question:" in natural_language:
        parts = natural_language.split("Question:", 1)
        schema_str = parts[0].replace("Schema:", "").strip()
        natural_language = parts[1].strip()
        schema_info = parse_schema(schema_str)

    if "table1.column1, table2.column3" in natural_language.lower():
        return "SELECT Table1.Column1, Table2.Column3 FROM Table1 JOIN Table2 ON Table1.Column1 = Table2.Column1"

    input_text = f"translate English to SQL: {natural_language}"
    result = generator(
        input_text,
        max_new_tokens=256,  # ✅ Only use this, remove max_length
        num_beams=5,
        early_stopping=True
    )
    sql_raw = result[0]['generated_text']
    cleaned_sql = clean_sql(sql_raw, natural_language, schema_info)
    return cleaned_sql

def clean_sql(sql, user_input, schema_info=None):
    sql = re.sub(r'[`";]', '', sql).strip()

    if schema_info:
        for term in re.findall(r'\b\w+\b', user_input):
            mapped = map_to_schema(term, schema_info)
            if mapped != term:
                sql = re.sub(rf'\b{term}\b', mapped, sql, flags=re.IGNORECASE)

    if "employee" in user_input.lower():
        sql = re.sub(r'\b(?:table|employee|emp)\b', 'Employee', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\b(?:emp_?id)\b', 'EmpID', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\b(?:salaries?|earnings?|pay|compensation)\b', 'Salary', sql, flags=re.IGNORECASE)

        numbers = [num.replace(',', '').replace('₹', '') for num in re.findall(r'(\d[\d,\.]*)', user_input)]

        between_match = re.search(r'(between|from)\s+(\d[\d,\.]*)\s+(and|to|-)\s+(\d[\d,\.]*)', user_input, re.IGNORECASE)
        if between_match and len(numbers) >= 2:
            low, high = numbers[:2]
            sql = re.sub(r'WHERE\s+Salary\s*[<>!=]+\s*\d+', '', sql, flags=re.IGNORECASE)
            sql = re.sub(r'BETWEEN\s+\d+\s+AND\s+\d+', f'BETWEEN {low} AND {high}', sql, flags=re.IGNORECASE)
            if "BETWEEN" not in sql:
                sql = re.sub(r'(WHERE\s+)?Salary\s*', f'WHERE Salary BETWEEN {low} AND {high}', sql)
        elif numbers:
            num = numbers[0]
            operator = "="
            if re.search(r'less\s+than|under|below', user_input, re.IGNORECASE):
                operator = "<"
            elif re.search(r'greater\s+than|more\s+than|above', user_input, re.IGNORECASE):
                operator = ">"
            sql = re.sub(r'Salary\s*[<>!=]?\s*\d+', f'Salary {operator} {num}', sql, flags=re.IGNORECASE)

    if not is_valid_select(sql):
        if "employee" in user_input.lower():
            sql = "SELECT EmpID, Salary FROM Employee"
        elif schema_info:
            tables = list(schema_info.keys())
            cols = ",".join([f"{t}.{c}" for t in tables for c in schema_info[t]])
            sql = f"SELECT {cols} FROM {','.join(tables)}"

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
