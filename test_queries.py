    from app.services.sqlgenerator import generate_sql

    # Test cases with schema context
    TEST_CASES = [
        ("Show employees earning more than 50000", 
        "SELECT EmpID, Salary FROM Employee WHERE Salary > 50000"),
        
        ("List all high earners", 
        "SELECT EmpID, Salary FROM Employee WHERE Salary > 100000"),
        
        ("Find employees with salaries between 40000 and 60000", 
        "SELECT EmpID, Salary FROM Employee WHERE Salary BETWEEN 40000 AND 60000"),
        
        ("Who are the top 10 highest paid employees?", 
        "SELECT EmpID, Salary FROM Employee ORDER BY Salary DESC LIMIT 10")
    ]

    def run_tests():
        print("Running SQL Generation Tests with Schema Context...")
        print("-" * 60)
        
        passed = 0
        for i, (query, expected) in enumerate(TEST_CASES):
            try:
                result = generate_sql(query)
                match = result.strip().lower() == expected.strip().lower()
                status = "PASS" if match else "FAIL"
                
                if match:
                    passed += 1
                
                print(f"Test {i+1}: {status}")
                print(f"Input:    {query}")
                print(f"Expected: {expected}")
                print(f"Actual:   {result}")
                print("-" * 60)
            except Exception as e:
                print(f"Test {i+1}: ERROR")
                print(f"Input: {query}")
                print(f"Error: {str(e)}")
                print("-" * 60)
        
        print(f"Results: {passed}/{len(TEST_CASES)} passed")
        print(f"Accuracy: {passed/len(TEST_CASES)*100:.2f}%")

    if __name__ == "__main__":
        run_tests()