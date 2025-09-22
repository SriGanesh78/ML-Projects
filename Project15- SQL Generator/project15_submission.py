#!/usr/bin/env python3
"""
Project 15 Submission: Employee Dataset for Natural Language to SQL
Alternative Python script version for when Jupyter renderer has issues
"""

import sqlite3
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 Project 15: Natural Language to SQL - Employee Dataset")
    print("=" * 60)
    
    # Step 1: Create Database Schema
    print("\n📊 Creating Employee Management System Database Schema...")
    
    conn = sqlite3.connect('employee_management.db')
    cursor = conn.cursor()
    
    # Departments table
    departments_schema = """
    CREATE TABLE IF NOT EXISTS departments (
        department_id INTEGER PRIMARY KEY,
        department_name VARCHAR(50) NOT NULL,
        manager_id INTEGER,
        budget DECIMAL(15,2),
        location VARCHAR(100),
        created_date DATE,
        FOREIGN KEY (manager_id) REFERENCES employees(employee_id)
    )
    """
    
    # Positions table
    positions_schema = """
    CREATE TABLE IF NOT EXISTS positions (
        position_id INTEGER PRIMARY KEY,
        position_title VARCHAR(100) NOT NULL,
        department_id INTEGER,
        min_salary DECIMAL(10,2),
        max_salary DECIMAL(10,2),
        job_level VARCHAR(20),
        requirements TEXT,
        FOREIGN KEY (department_id) REFERENCES departments(department_id)
    )
    """
    
    # Employees table
    employees_schema = """
    CREATE TABLE IF NOT EXISTS employees (
        employee_id INTEGER PRIMARY KEY,
        first_name VARCHAR(50) NOT NULL,
        last_name VARCHAR(50) NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        phone VARCHAR(20),
        hire_date DATE NOT NULL,
        position_id INTEGER,
        manager_id INTEGER,
        salary DECIMAL(10,2),
        status VARCHAR(20) DEFAULT 'active',
        address TEXT,
        city VARCHAR(50),
        state VARCHAR(50),
        zip_code VARCHAR(10),
        birth_date DATE,
        gender VARCHAR(10),
        FOREIGN KEY (position_id) REFERENCES positions(position_id),
        FOREIGN KEY (manager_id) REFERENCES employees(employee_id)
    )
    """
    
    # Salaries table
    salaries_schema = """
    CREATE TABLE IF NOT EXISTS salaries (
        salary_id INTEGER PRIMARY KEY,
        employee_id INTEGER,
        salary_amount DECIMAL(10,2) NOT NULL,
        effective_date DATE NOT NULL,
        end_date DATE,
        salary_type VARCHAR(20) DEFAULT 'base',
        bonus DECIMAL(10,2) DEFAULT 0,
        FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
    )
    """
    
    # Performance Reviews table
    performance_reviews_schema = """
    CREATE TABLE IF NOT EXISTS performance_reviews (
        review_id INTEGER PRIMARY KEY,
        employee_id INTEGER,
        review_date DATE NOT NULL,
        reviewer_id INTEGER,
        rating DECIMAL(3,1),
        goals_achieved INTEGER,
        goals_total INTEGER,
        comments TEXT,
        next_review_date DATE,
        FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
        FOREIGN KEY (reviewer_id) REFERENCES employees(employee_id)
    )
    """
    
    # Execute schema creation
    schemas = [
        ("Departments", departments_schema),
        ("Positions", positions_schema),
        ("Employees", employees_schema),
        ("Salaries", salaries_schema),
        ("Performance Reviews", performance_reviews_schema)
    ]
    
    for table_name, schema in schemas:
        cursor.execute(schema)
        print(f"✅ {table_name} table created successfully!")
    
    # Step 2: Generate Sample Data
    print("\n👥 Generating realistic employee data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Sample data
    first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda',
                   'William', 'Elizabeth', 'David', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica']
    
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                  'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas']
    
    departments_data = [
        ('Human Resources', 1500000, 'Building A, Floor 2', '2020-01-15'),
        ('Engineering', 5000000, 'Building B, Floor 3-5', '2019-06-01'),
        ('Sales', 3000000, 'Building A, Floor 1', '2019-03-10'),
        ('Marketing', 2000000, 'Building C, Floor 2', '2020-02-20'),
        ('Finance', 1200000, 'Building A, Floor 3', '2019-01-01'),
        ('Operations', 2500000, 'Building D, Floor 1-2', '2019-09-15'),
        ('Customer Support', 1800000, 'Building C, Floor 1', '2020-04-01'),
        ('Research & Development', 4000000, 'Building E, Floor 2-4', '2019-11-01')
    ]
    
    positions_data = [
        # HR Positions
        ('HR Manager', 1, 80000, 120000, 'Senior', 'Bachelor degree in HR, 5+ years experience'),
        ('HR Specialist', 1, 50000, 70000, 'Mid', 'Bachelor degree, 2+ years experience'),
        ('Recruiter', 1, 45000, 65000, 'Mid', 'Bachelor degree, 1+ years experience'),
        
        # Engineering Positions
        ('Senior Software Engineer', 2, 100000, 150000, 'Senior', 'Computer Science degree, 5+ years'),
        ('Software Engineer', 2, 70000, 110000, 'Mid', 'Computer Science degree, 2+ years'),
        ('Junior Developer', 2, 50000, 75000, 'Junior', 'Computer Science degree or bootcamp'),
        
        # Sales Positions
        ('Sales Director', 3, 120000, 180000, 'Senior', 'MBA preferred, 8+ years sales'),
        ('Sales Manager', 3, 80000, 120000, 'Senior', 'Bachelor degree, 5+ years sales'),
        ('Sales Representative', 3, 45000, 80000, 'Mid', 'Bachelor degree, 1+ years sales'),
        
        # Marketing Positions
        ('Marketing Manager', 4, 70000, 100000, 'Senior', 'Marketing degree, 4+ years experience'),
        ('Digital Marketing Specialist', 4, 50000, 75000, 'Mid', 'Marketing degree, 2+ years'),
        
        # Finance Positions
        ('CFO', 5, 150000, 250000, 'Executive', 'CPA, MBA, 10+ years finance'),
        ('Financial Analyst', 5, 60000, 90000, 'Mid', 'Finance degree, 2+ years experience'),
        ('Accountant', 5, 45000, 65000, 'Mid', 'Accounting degree, CPA preferred'),
        
        # Operations Positions
        ('Operations Manager', 6, 80000, 120000, 'Senior', 'Business degree, 5+ years ops'),
        ('Project Manager', 6, 70000, 100000, 'Senior', 'PMP certification preferred'),
        
        # Customer Support Positions
        ('Support Manager', 7, 60000, 85000, 'Senior', 'Bachelor degree, 4+ years support'),
        ('Customer Success Specialist', 7, 45000, 65000, 'Mid', 'Bachelor degree, 2+ years'),
        
        # R&D Positions
        ('Research Director', 8, 130000, 180000, 'Senior', 'PhD in relevant field, 8+ years'),
        ('Research Scientist', 8, 80000, 120000, 'Senior', 'PhD or Masters, 3+ years research')
    ]
    
    # Insert departments and positions
    for i, (name, budget, location, created_date) in enumerate(departments_data, 1):
        cursor.execute("""
            INSERT INTO departments (department_id, department_name, budget, location, created_date)
            VALUES (?, ?, ?, ?, ?)
        """, (i, name, budget, location, created_date))
    
    for i, (title, dept_id, min_sal, max_sal, level, requirements) in enumerate(positions_data, 1):
        cursor.execute("""
            INSERT INTO positions (position_id, position_title, department_id, min_salary, max_salary, job_level, requirements)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (i, title, dept_id, min_sal, max_sal, level, requirements))
    
    # Step 3: Generate Employee Records
    print("👤 Generating employee records...")
    
    num_employees = 50  # Reduced for faster execution
    employees = []
    
    for i in range(1, num_employees + 1):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        email = f"{first_name.lower()}.{last_name.lower()}{i}@company.com"
        phone = f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"
        
        hire_date = datetime(2018, 1, 1) + timedelta(days=random.randint(0, 2000))
        
        position_id = random.randint(1, len(positions_data))
        manager_id = random.randint(1, min(i-1, 10)) if i > 10 and random.random() < 0.7 else None
        
        position_info = positions_data[position_id - 1]
        salary = random.uniform(position_info[2], position_info[3])
        
        status = 'active' if random.random() > 0.05 else 'inactive'
        
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
        states = ['NY', 'CA', 'IL', 'TX', 'AZ']
        
        city = random.choice(cities)
        state = random.choice(states)
        address = f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'First'])} St"
        zip_code = f"{random.randint(10000, 99999)}"
        
        birth_year = random.randint(1959, 2002)
        birth_date = f"{birth_year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        gender = random.choice(['Male', 'Female', 'Other'])
        
        employees.append((
            i, first_name, last_name, email, phone, hire_date.strftime('%Y-%m-%d'),
            position_id, manager_id, round(salary, 2), status, address, city, state,
            zip_code, birth_date, gender
        ))
    
    # Insert employees
    for employee in employees:
        cursor.execute("""
            INSERT INTO employees (employee_id, first_name, last_name, email, phone, hire_date,
                                 position_id, manager_id, salary, status, address, city, state,
                                 zip_code, birth_date, gender)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, employee)
    
    # Step 4: Demonstrate NL2SQL Queries
    print("\n🎯 Demonstrating Natural Language to SQL Queries...")
    
    def execute_query(question, sql, description):
        print(f"\n🔍 Question: {question}")
        print(f"📝 SQL: {sql}")
        print(f"💡 Description: {description}")
        
        try:
            result = pd.read_sql_query(sql, conn)
            print(f"📊 Results ({len(result)} rows):")
            print(result.head().to_string(index=False))
            if len(result) > 5:
                print(f"... and {len(result) - 5} more rows")
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 50)
    
    # Query Examples
    queries = [
        ("Show me all employees in the Engineering department", """
            SELECT e.first_name, e.last_name, e.email, p.position_title, e.salary
            FROM employees e
            JOIN positions p ON e.position_id = p.position_id
            JOIN departments d ON p.department_id = d.department_id
            WHERE d.department_name = 'Engineering'
            ORDER BY e.salary DESC
        """, "Retrieves engineering employees with their positions and salaries"),
        
        ("Who are the highest paid employees?", """
            SELECT e.first_name, e.last_name, p.position_title, d.department_name, e.salary
            FROM employees e
            JOIN positions p ON e.position_id = p.position_id
            JOIN departments d ON p.department_id = d.department_id
            WHERE e.status = 'active'
            ORDER BY e.salary DESC
            LIMIT 5
        """, "Shows top 5 highest paid active employees"),
        
        ("Show department budget summary", """
            SELECT d.department_name, COUNT(e.employee_id) as employee_count,
                   ROUND(AVG(e.salary), 2) as avg_salary, d.budget
            FROM departments d
            LEFT JOIN positions p ON d.department_id = p.department_id
            LEFT JOIN employees e ON p.position_id = e.position_id AND e.status = 'active'
            GROUP BY d.department_id, d.department_name, d.budget
            ORDER BY employee_count DESC
        """, "Provides budget analysis by department")
    ]
    
    for question, sql, description in queries:
        execute_query(question, sql, description)
    
    # Final Summary
    print("\n📊 FINAL SUMMARY")
    print("=" * 30)
    
    stats = [
        ("Total Employees", "SELECT COUNT(*) FROM employees"),
        ("Active Employees", "SELECT COUNT(*) FROM employees WHERE status = 'active'"),
        ("Departments", "SELECT COUNT(*) FROM departments"),
        ("Average Salary", "SELECT ROUND(AVG(salary), 2) FROM employees WHERE status = 'active'")
    ]
    
    for stat_name, query in stats:
        result = cursor.execute(query).fetchone()
        print(f"📈 {stat_name}: {result[0]}")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Database created: employee_management.db")
    print(f"🎉 Project 15 submission completed successfully!")
    print(f"\n💡 If you still have Jupyter renderer issues, you can:")
    print(f"   1. Run this Python script instead: python project15_submission.py")
    print(f"   2. Use Jupyter Lab in your browser")
    print(f"   3. Use Google Colab for the notebook")

if __name__ == "__main__":
    main()
