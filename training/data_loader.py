def generate_synthetic_data():
    """Optional: Generate more training examples"""
    base_phrases = [
        "Show employees earning more than {value}",
        "List staff with salary above {value}",
        "Display workers making over {value}",
        "Find employees with salaries greater than {value}",
        "Get records where pay exceeds {value}"
    ]
    
    salaries = [5000, 10000, 15000, 20000, 25000, 30000]
    
    with open('../data/training_data.jsonl', 'a') as f:
        for phrase in base_phrases:
            for salary in salaries:
                input_text = f"translate English to SQL: {phrase.format(value=salary)}"
                target_sql = f"SELECT EmpID, Salary FROM Employee WHERE Salary > {salary}"
                f.write(f'{{"input": "{input_text}", "target": "{target_sql}"}}\n')