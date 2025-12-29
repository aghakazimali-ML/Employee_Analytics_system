import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Phase 1: Load data
df = pd.read_csv("data/employee_analytics_messy_10000.csv")
print(f"Initial Shape: {df.shape}")

# Phase 2: Handle Duplicates
df = df.drop_duplicates()

# Phase 3: Clean Departments
df["Department"] = df["Department"].fillna("Unknown")

# Phase 4: Clean Salaries and Performance
# Step A: Identify salaries that are 0 or negative (Invalid)
invalid_salaries_filter = (df["Salary"] <= 0) 
df.loc[invalid_salaries_filter, "Salary"] = np.nan

# Step B: Calculate medians (with parentheses!)
salary_median = df['Salary'].median()
perf_median = df['Performance_Score'].median()

# Step C: Fill the NaNs
df['Salary'] = df['Salary'].fillna(salary_median)
df['Performance_Score'] = df["Performance_Score"].fillna(perf_median)

# Ensure data types are numeric for analysis
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')

print("\n--- Final Data Readiness Report ---")
print(f"Final Shape: {df.shape}")
print("Missing values remaining:\n", df.isnull().sum())

# Save the cleaned data
df.to_csv("employee_analytics_cleaned.csv", index=False)
print("\nSuccess! 'employee_analytics_cleaned.csv' is ready.")

# Phase 5: Grouped Analysis
dept_stats = df.groupby('Department')['Salary'].agg(['mean', 'median', 'min', 'max', 'count', 'sum'])
print("\nDepartment Statistics (Sorted by Mean Salary):")
print(dept_stats.sort_values(by='mean', ascending=False))

# Visualization 1: Salary Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['Salary'], bins=20, kde=True, color="skyblue")
plt.title("Company-Wide Salary Distribution")
plt.xlabel("Salary ($)")
plt.ylabel("Number of Employees")
plt.savefig("salaries_distribution.png")
plt.show()

# Visualization 2: Pay Gaps by Department
plt.figure(figsize=(12, 6))
# Updated syntax to avoid warnings
sns.boxplot(data=df, x='Department', y='Salary', hue='Department', palette='Set2', legend=False)
plt.title("Visualizing Pay Gaps by Department")
plt.xticks(rotation=45)
plt.tight_layout() # Prevents labels from being cut off
plt.savefig("department_pay_range.png")
plt.show()

## Phase 6: Department Insights Module
#**Second major analytics feature**

dept_insights = df.groupby('Department').agg(
    Headcount=('Salary', 'count'),
    Avg_Salary=('Salary', 'mean'),
    Total_Cost=('Salary', 'sum'),
    Avg_Performance=('Performance_Score', 'mean')
).reset_index()

total_budget = dept_insights['Total_Cost'].sum()
dept_insights['Cost_Contribution_%'] = (dept_insights['Total_Cost'] / total_budget) * 100
dept_insights['Efficiency_Ratio'] = dept_insights['Avg_Performance'] / (dept_insights['Avg_Salary'] / 1000)

print("\n--- Department Comparison Report ---")
print(dept_insights.sort_values(by='Total_Cost', ascending=False).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.scatterplot(data=dept_insights, x='Headcount', y='Total_Cost', size='Avg_Salary', hue='Department', sizes=(100, 1000))
plt.title("Headcount vs. Total Cost")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("dept_analysis_scatter.png")
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(dept_insights['Total_Cost'], labels=dept_insights['Department'], autopct='%1.1f%%')
plt.title("Budget Distribution by Department")
plt.savefig("budget_pie_chart.png")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=dept_insights.sort_values('Efficiency_Ratio'), x='Department', y='Efficiency_Ratio', palette='magma')
plt.title("Department Efficiency (Performance per $1k Spent)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("dept_efficiency.png")
plt.show()

## Phase 7: Employee Ranking System

# 1. Composite Score Calculation
# Note: I standardized the name to 'Composite_Score' to avoid typos
df["Composite_Score"] = (df["Performance_Score"] * 0.7) + ((df['Salary'] / df['Salary'].max()) * 0.3)

# 2. Organization-Wide Ranking
df['Org_Rank'] = df['Composite_Score'].rank(ascending=False, method='min')

# 3. Department-wide ranking
df['Dept_Rank'] = df.groupby('Department')['Composite_Score'].rank(ascending=False, method='min')

# 4. Identify Top Performers
# FIX: Use df[df['Column'] <= 10] to filter the table first, then sort
top_performer = df[df['Org_Rank'] <= 10].sort_values(by='Org_Rank')

# 5. Identify Underperformers
# FIX: Use square brackets [] for filtering, not parentheses ()
underperformer = df[df['Performance_Score'] < df['Performance_Score'].quantile(0.10)]

print("\n--- Phase 7: Top 10 Employees (Organization-Wide) ---")
# FIX: Ensure column names match exactly (Org_Rank vs Org_rank)
print(top_performer[['Department', 'Salary', 'Performance_Score', 'Org_Rank']])

# Save to CSV
top_performer.to_csv("top_performers_ranking.csv", index=False)

#Phase 8 :  Visualization & Reporting


# 1. Salary Distribution by Department (Boxplot)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Department', y='Salary', hue='Department', palette='viridis', legend=False)
plt.title("Salary Range and Outliers by Department", fontsize=14)
plt.xlabel("Department", fontsize=12)
plt.ylabel("Salary ($)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("salary_distribution_final.png")
plt.show()

# 2. Performance vs. Salary Correlation (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Salary', y='Performance_Score', hue='Department', alpha=0.6, s=80)
plt.title("Performance Score vs. Salary Level", fontsize=14)
plt.axvline(df['Salary'].median(), color='red', linestyle='--', label='Median Salary')
plt.legend()
plt.tight_layout()
plt.savefig("performance_v_salary.png")
plt.show()

# 3. Departmental Cost Contribution (Pie Chart)
plt.figure(figsize=(8, 8))
dept_costs = df.groupby('Department')['Salary'].sum()
plt.pie(dept_costs, labels=dept_costs.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title("Total Budget Share per Department", fontsize=14)
plt.savefig("budget_share_pie.png")
plt.show()

# 4. Top 10 Employees Ranking (Horizontal Bar Chart)
plt.figure(figsize=(10, 6))
top_10 = df.nsmallest(10, 'Org_Rank')
sns.barplot(data=top_10, x='Composite_Score', y='Department', hue='Department', palette='magma', legend=False)
plt.title("Top 10 Employees: Composite Ranking Score", fontsize=14)
plt.xlabel("Composite Score (Weighted)", fontsize=12)
plt.tight_layout()
plt.savefig("top_rankings_bar.png")
plt.show()

## Phase 9: Insight Interpretation

# 1. Define the baseline for "Average Performance"
avg_performance = df['Performance_Score'].mean()

# 2. Identify "Underpaid Stars" 
# FIXED: Changed df(...) to df[...] and added & logic inside brackets
underpaid_stars = df[(df['Performance_Score'] > 8) & (df['Salary'] < df['Salary'].median())]

# 3. Identify High-Cost vs. Low-Performance Departments
high_cost_low_perf_dept = dept_insights[
    (dept_insights['Cost_Contribution_%'] > 20) & (dept_insights['Avg_Performance'] < avg_performance)
]

# 2. Answer Business Questions & Highlight Risks
print("\n" + "="*45)
print("--- PHASE 9: EXECUTIVE INSIGHT NARRATIVE ---")
print("="*45)

print("\n[RISK: Retention Alert]")
# FIXED: variable name now matches 'underpaid_stars'
print(f"Found {len(underpaid_stars)} employees with high performance but below-average pay.")
print("Recommendation: Review compensation for these individuals to prevent turnover.")

print("\n[OPPORTUNITY: Budget Optimization]")
if not high_cost_low_perf_dept.empty:
    for _, row in high_cost_low_perf_dept.iterrows():
        print(f"The '{row['Department']}' department has high costs but below-average performance.")
    print("Recommendation: Audit processes in these departments to improve ROI.")
else:
    print("No immediate departmental budget risks identified.")

# 3. Final Recommendations
print("\n[STRATEGIC RECOMMENDATIONS]")
print("1. Performance-Pay Alignment: Increase the correlation between score and salary.")
print("2. Resource Reallocation: Move budget from 'High-Cost' to 'High-Value' departments.")
print("3. Training Focus: Identify the bottom 10% of performers for a 90-day improvement plan.")

