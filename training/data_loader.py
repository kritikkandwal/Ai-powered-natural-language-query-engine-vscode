def generate_synthetic_data():
    import random, json
    base_phrases = [
        "Show employees earning more than {value}",
        "List employees with salary above {value}",
        "Display staff earning above {value}",
        "Find employees earning over {value}",
        "Who earns greater than {value}"
    ]
    salaries = [3000, 5000, 7000, 10000, 15000, 20000, 25000, 30000]

    with open('../data/training_data.jsonl', 'w') as f:
        for salary in salaries:
            for template in base_phrases:
                input_text = f"translate English to SQL: {template.format(value=salary)}"
                target_sql = f"SELECT EmpID, Salary FROM Employee WHERE Salary > {salary}"
                f.write(json.dumps({"input": input_text, "target": target_sql}) + "\n")

if __name__ == "__main__":
    generate_synthetic_data()
    print("✅ Synthetic training data generated!")