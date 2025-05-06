import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of records to generate
n_legitimate = 9000  # 90% legitimate records
n_fraudulent = 1000  # 10% fraudulent records
n_total = n_legitimate + n_fraudulent

# Generate basic taxpayer information
def generate_taxpayer_data(n):
    data = {
        'taxpayer_id': [f'TP{i:06d}' for i in range(1, n+1)],
        'age': np.random.randint(18, 85, n),
        'income_reported': np.random.lognormal(mean=11, sigma=0.7, size=n),  # Log-normal distribution for income
        'deductions_claimed': np.zeros(n),
        'tax_credits_claimed': np.zeros(n),
        'previous_audit_flag': np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        'years_filing': np.random.randint(1, 40, n),
        'num_dependents': np.random.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03]),
        'self_employed': np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
        'foreign_income': np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        'rental_income': np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        'investment_income': np.zeros(n),
        'business_expenses': np.zeros(n),
        'charitable_contributions': np.zeros(n),
        'medical_expenses': np.zeros(n),
        'education_expenses': np.zeros(n),
        'mortgage_interest': np.zeros(n),
        'property_tax': np.zeros(n),
        'state_local_tax': np.zeros(n),
        'filing_status': np.random.choice(['Single', 'Married', 'Head of Household'], size=n, p=[0.45, 0.45, 0.1]),
        'occupation_category': np.random.choice(['Professional', 'Service', 'Manual Labor', 'Management', 'Technical', 'Sales', 'Administrative', 'Self-employed'], size=n),
        'industry_sector': np.random.choice(['Healthcare', 'Technology', 'Finance', 'Education', 'Retail', 'Manufacturing', 'Construction', 'Transportation', 'Hospitality', 'Other'], size=n),
        'geographic_region': np.random.choice(['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West'], size=n),
        'filing_method': np.random.choice(['Electronic', 'Paper'], size=n, p=[0.85, 0.15]),
        'filing_date': [(datetime(2023, 1, 1) + timedelta(days=random.randint(0, 105))).strftime('%Y-%m-%d') for _ in range(n)],
    }
    
    # Calculate dependent values
    for i in range(n):
        # Investment income based on reported income
        data['investment_income'][i] = data['income_reported'][i] * np.random.uniform(0, 0.2)
        
        # Deductions and expenses as percentages of income
        income = data['income_reported'][i]
        data['business_expenses'][i] = income * np.random.uniform(0, 0.3) if data['self_employed'][i] else 0
        data['charitable_contributions'][i] = income * np.random.uniform(0, 0.1)
        data['medical_expenses'][i] = income * np.random.uniform(0, 0.05)
        data['education_expenses'][i] = income * np.random.uniform(0, 0.05)
        data['mortgage_interest'][i] = income * np.random.uniform(0, 0.2)
        data['property_tax'][i] = income * np.random.uniform(0, 0.05)
        data['state_local_tax'][i] = income * np.random.uniform(0.02, 0.1)
        
        # Total deductions
        data['deductions_claimed'][i] = (data['business_expenses'][i] + 
                                        data['charitable_contributions'][i] + 
                                        data['medical_expenses'][i] + 
                                        data['education_expenses'][i] + 
                                        data['mortgage_interest'][i] + 
                                        data['property_tax'][i] + 
                                        data['state_local_tax'][i])
        
        # Tax credits based on dependents and income
        data['tax_credits_claimed'][i] = data['num_dependents'][i] * 2000 + np.random.uniform(0, 5000) if income < 100000 else data['num_dependents'][i] * 1000
    
    return pd.DataFrame(data)

# Generate legitimate taxpayer records
legitimate_data = generate_taxpayer_data(n_legitimate)
legitimate_data['fraud_flag'] = 0

# Generate fraudulent taxpayer records with anomalies
fraudulent_data = generate_taxpayer_data(n_fraudulent)

# Introduce fraud patterns
for i in range(n_fraudulent):
    fraud_type = np.random.choice(['income_underreporting', 'deduction_inflation', 'credit_abuse', 'identity_theft', 'mixed'])
    
    if fraud_type == 'income_underreporting' or fraud_type == 'mixed':
        # Underreport income by 20-50%
        true_income = fraudulent_data.loc[i, 'income_reported']
        fraudulent_data.loc[i, 'income_reported'] = true_income * np.random.uniform(0.5, 0.8)
    
    if fraud_type == 'deduction_inflation' or fraud_type == 'mixed':
        # Inflate deductions by 50-200%
        for deduction in ['business_expenses', 'charitable_contributions', 'medical_expenses', 'education_expenses', 'mortgage_interest', 'property_tax']:
            fraudulent_data.loc[i, deduction] *= np.random.uniform(1.5, 3.0)
        
        # Recalculate total deductions
        fraudulent_data.loc[i, 'deductions_claimed'] = (fraudulent_data.loc[i, 'business_expenses'] + 
                                                      fraudulent_data.loc[i, 'charitable_contributions'] + 
                                                      fraudulent_data.loc[i, 'medical_expenses'] + 
                                                      fraudulent_data.loc[i, 'education_expenses'] + 
                                                      fraudulent_data.loc[i, 'mortgage_interest'] + 
                                                      fraudulent_data.loc[i, 'property_tax'] + 
                                                      fraudulent_data.loc[i, 'state_local_tax'])
    
    if fraud_type == 'credit_abuse' or fraud_type == 'mixed':
        # Claim excessive tax credits
        fraudulent_data.loc[i, 'tax_credits_claimed'] *= np.random.uniform(1.5, 2.5)
    
    if fraud_type == 'identity_theft':
        # Completely fabricated return with unusual patterns
        fraudulent_data.loc[i, 'filing_method'] = 'Electronic'
        fraudulent_data.loc[i, 'filing_date'] = (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d')  # Early filing
        fraudulent_data.loc[i, 'previous_audit_flag'] = 0
        fraudulent_data.loc[i, 'tax_credits_claimed'] = fraudulent_data.loc[i, 'income_reported'] * np.random.uniform(0.1, 0.3)

fraudulent_data['fraud_flag'] = 1

# Combine legitimate and fraudulent data
combined_data = pd.concat([legitimate_data, fraudulent_data], ignore_index=True)

# Calculate additional features
combined_data['deduction_to_income_ratio'] = combined_data['deductions_claimed'] / combined_data['income_reported']
combined_data['credit_to_income_ratio'] = combined_data['tax_credits_claimed'] / combined_data['income_reported']
combined_data['days_to_deadline'] = combined_data['filing_date'].apply(lambda x: (datetime.strptime('2023-04-15', '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d')).days)

# Add some derived features that might be useful for detection
combined_data['expense_per_dependent'] = combined_data.apply(
    lambda row: row['deductions_claimed'] / (row['num_dependents'] + 1), axis=1
)
combined_data['income_per_dependent'] = combined_data.apply(
    lambda row: row['income_reported'] / (row['num_dependents'] + 1), axis=1
)

# Shuffle the data
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
combined_data.to_csv('synthetic_tax_fraud_dataset.csv', index=False)

# Create a data dictionary
data_dictionary = pd.DataFrame({
    'Feature': combined_data.columns,
    'Description': [
        'Unique identifier for the taxpayer',
        'Age of the taxpayer',
        'Total income reported on tax return',
        'Total deductions claimed',
        'Total tax credits claimed',
        'Flag indicating if the taxpayer was audited in the past (1=Yes, 0=No)',
        'Number of years the taxpayer has been filing taxes',
        'Number of dependents claimed',
        'Flag indicating if the taxpayer is self-employed (1=Yes, 0=No)',
        'Flag indicating if the taxpayer reported foreign income (1=Yes, 0=No)',
        'Flag indicating if the taxpayer reported rental income (1=Yes, 0=No)',
        'Income from investments (interest, dividends, capital gains)',
        'Business expenses claimed by self-employed taxpayers',
        'Charitable contributions claimed',
        'Medical expenses claimed',
        'Education-related expenses claimed',
        'Mortgage interest deduction claimed',
        'Property tax deduction claimed',
        'State and local tax deduction claimed',
        'Filing status (Single, Married, Head of Household)',
        'Category of the taxpayer\'s occupation',
        'Industry sector of the taxpayer\'s employment',
        'Geographic region of the taxpayer',
        'Method used to file taxes (Electronic or Paper)',
        'Date when the tax return was filed',
        'Fraud flag (1=Fraudulent, 0=Legitimate)',
        'Ratio of deductions claimed to income reported',
        'Ratio of tax credits claimed to income reported',
        'Number of days between filing date and tax deadline',
        'Deductions claimed per dependent (including the taxpayer)',
        'Income reported per dependent (including the taxpayer)'
    ],
    'Type': [str(dtype) for dtype in combined_data.dtypes]
})

data_dictionary.to_csv('data_dictionary.csv', index=False)

print(f"Generated dataset with {n_total} records ({n_fraudulent} fraudulent, {n_legitimate} legitimate)")
print(f"Dataset saved as 'synthetic_tax_fraud_dataset.csv'")
print(f"Data dictionary saved as 'data_dictionary.csv'")

# Display some statistics
print("\nDataset Statistics:")
print(f"Fraud rate: {combined_data['fraud_flag'].mean()*100:.2f}%")
print("\nFeature statistics:")
print(combined_data.describe().transpose())
