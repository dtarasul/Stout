#!/usr/bin/env python
# coding: utf-8

# In[80]:


import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statsmodels.api as sm
import numpy as np
import math 
import seaborn as sns


# In[145]:


loans = pd.read_csv("loans_full_schema.csv")


# In[41]:


print("Loans is a dataset (converted to a dataframe) with 10,000 rows and 55 columns. Each row is an observation that represents a loan made through the lending club, and then each of the column is some piece of information about that loan. There is a litany of columns that are descriptors of the applicant, from information on their income to the number of credit lines they have open, and whether it is an individual or joint applicant. Then there is a series of columns about the loanitself, specifying the loan amount, term, grade, how much has been paid by the applicant, interest rate, and more. Some initial issues is that the issue months aren't normalized. Another data issue is that there are some cells that are fully missing data, and some that say NA,which need to be treated differently because NA is true, for example, if a loan is individual but the column is for joint income, but if there is missing data altogether, that needs to be cleaned up. Another issue is that there are no units listed in either the documentation or the data itself, so I will be making assumptions, such that the monetary values are in dollars.")


# In[21]:


loans


# In[57]:


plt.scatter(x = loans.dropna()["annual_income"], y =  loans.dropna()["interest_rate"], c = loans.dropna()["total_credit_limit"], cmap=cm.inferno)

plt.title("Relationship Between Annual Income and Interest Rate on Loans")
plt.xlabel("Annual Income (Dollars)")
plt.ylabel("Loan Interest Rate (Percent)")
cbar = plt.colorbar()
cbar.set_label('Total Credit Limit (Dollars)')
plt.show()


# In[64]:


plt.scatter(x = loans.where(loans.annual_income < 150000).dropna()["annual_income"], y =  loans.where(loans.annual_income < 150000).dropna()["interest_rate"], c = -loans.where(loans.annual_income < 150000).dropna()["total_credit_limit"], cmap=cm.inferno)

plt.title("Relationship Between Annual Income and Interest Rate on Loans")
plt.xlabel("Annual Income (Dollars)")
plt.ylabel("Loan Interest Rate (Percent)")
cbar = plt.colorbar()
cbar.set_label('(Negative) Total Credit Limit, Dollars')
plt.show()


# In[65]:


print("To begin my exploratory analysis through data visualization I wanted to see if there was a meaningful relationship between annual income and interest rate, as well as adding in the dimension of total credit limit. The relationship isn't always clear, but it appears that as annual income increases, loan interest rate definitely trends downwards (as noted by the lack of data points in the upper right hand corner of the graph.) There are two versions of this scatter plot. The first has the whole set of data with no limits on annual income. I noticed however, that the plot got very sparse on the far right side, and decided to cut those outliers out to get a better understanding. Additionally, I flipped the values of the total credit limit (so they are negative now), in order for it to be more visually appealing. From this graph we can also see that lower loan interest rates also generally correspond to higher total credit limits, as well as annual income and totaly credit limit having a positive direct relationship.")


# In[72]:


loans.dropna().boxplot(by=['grade'], column=['loan_amount'])
title_boxplot = "Boxplot of Loan Amounts Grouped by Grade"
plt.title( title_boxplot )
plt.suptitle('')
plt.ylabel("Loan Amount (USD)") 
plt.xlabel("Loan Grade ") 


# In[73]:


print("There is no well defined relationship between a loan grade and its corresponding median loan amounts. However, highly graded loans are more widely distributed, and contain the highest echelon of loan amounts. This decreases as grades worsen, except for an interesting discrepancy where loans graded E go up. Loans graded F have the narrowest distribution, which means the vast majority of E loans are 20,000 dollars.")


# In[76]:


loans.hist(column = "debt_to_income", bins = 100)
plt.title("Distribution of Debt to Income Ratios")
plt.xlabel("Debt to Income Ratio")
plt.ylabel("Number of Loans")
plt.show()


# In[78]:


loans.where(loans.debt_to_income < 60).hist(column = "debt_to_income", bins = 100)
plt.title("Distribution of Debt to Income Ratios Under 60")
plt.xlabel("Debt to Income Ratio")
plt.ylabel("Number of Loans")
plt.show()


# In[79]:


print("This histogram gives some insight on typical values of debt to income ratios, as well as the shape of their distribution. Debt to income is monthly debt payments divided by monthly income, and the higher it is the less likely a borrower is to smoothly make all their monthly payments on their loan. It looks like the mode of this ratio is around 20, and approximately normally distributed around 20. 40 is where the distribution tapers off, which also happens to be the approximate cutoff for getting a qualified mortgage.")


# In[85]:


plt.figure(figsize=(15,8))

sns.countplot(loans['loan_status'])
plt.title("Counts of Loans By Status")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.show()


# In[86]:


print("This countplot gives some insight into the stages at which the loans which this data describes currently are. The vast majority of the loans have a status of current, so they are in progress and not late, a couple hundred are fully paid, and very small amounts are in the grace period, late, or charged off.")


# In[128]:


plt.scatter(x = loans.dropna()["months_since_last_delinq"], y =  loans.dropna()["term"])

plt.title("Relationship Between Delinquencies and the Length of Term granted on the loan")
plt.xlabel("Months Since Last Delinquency")
plt.ylabel("Loan Term")

plt.show()


# In[93]:


loans.hist(column = "term", bins = 30)
plt.title("Loan Term Distribution")
plt.xlabel("Term Length")
plt.ylabel("Loan Count")

plt.show()


# In[ ]:


print("I was interested to see if there was a relationship between loan delinquencies, and I chose the variable of interest to be months since the last delinquency, and the length of term for the loan granted to the borrower. I was surprised to see how non-continuous it was, and decided to make a histogram to double check the distribution of these term lengths. It looks like it is just 35 or 60 (months?), so if this becomes a predictive variable it is functionally binary.")


# In[169]:


#question 3 - creating a feature set and models to predict interest rate
print("I began by splitting up the data into test and train data, and putting the feature set columns and the target (interest rate) into separate dataframes. I also cleaned the data to remove all rows with missing information.")

print("My first model of interest is a linear regression. This requires numeric inputs. There are many categorical variables in the dataset - some of which lend themselves conversion to numerical values such as the inherent ordinality of the grades associated with loans, but some do not, such as the state where the applicant is or the employee title (although there are creative ways to play with the latter, such as assigning prestige scores or something of the sort). I will also exlude the information that post-dates the interest rate decision, such as the current balance. There is a lot to explore in these respects, but for initial simplicity of these models I will stick to already numeric variables. Insights from categorical variables can be abundant and should be explored in future analysis. I will also only perform this model on individual applications. It can obviously be repeated on joint applications or even combined.")

print("It is worthwhile to note here that I didn't drop rows with missing values, because that only leaves me with 200 observations, so instead I just replaced it with empty strings")
loans2 = loans[loans.application_type == "individual"]
loans4 = loans2.replace()
datadf = loans4[["emp_length", "annual_income", "debt_to_income", "delinq_2y", "months_since_last_delinq", "earliest_credit_line", "inquiries_last_12m", "total_credit_lines", "open_credit_lines", "total_credit_limit", "total_credit_utilized", "num_collections_last_12m", "num_historical_failed_to_pay", "months_since_90d_late", "current_accounts_delinq", "total_collection_amount_ever", "current_installment_accounts", "accounts_opened_24m", "months_since_last_credit_inquiry", "num_satisfactory_accounts", "num_accounts_120d_past_due", "num_accounts_30d_past_due", "num_active_debit_accounts", "total_debit_limit", "num_total_cc_accounts", "num_open_cc_accounts", "num_cc_carrying_balance", "num_mort_accounts", "account_never_delinq_percent", "tax_liens", "public_record_bankrupt", "loan_amount", "term" ]].dropna()
targetdf = loans4["interest_rate"].dropna()


# In[168]:


loans4


# In[170]:


#splitting into train and test
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(datadf, targetdf, test_size=0.3)


# In[171]:


#training and testing a linear regression
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(Xtrain,Ytrain)
linear_model.score(Xtest,Ytest)


# In[172]:


print("The linear regression is not particularly good at predicting the interest rate. It has a 34.4% accuracy. Improvements to this model would include normalizing the values of all the features (like placing them between 0 and 1) so their respective magnitudes don't skew the results")


# In[173]:


print("If we were to use this model to predict a new interest rate we would use the following variable coefficients, lining up with the order of features in teh feature set, with the following intercept, to get a predicted value for the interest rate")

print("coefs:", linear_model.coef_)
print("intercept:", linear_model.intercept_)


# In[175]:


from sklearn.ensemble import GradientBoostingRegressor
gbt_model = GradientBoostingRegressor()
gbt_model.fit(Xtrain, Ytrain)
print("Accuracy:", gbt_model.score(Xtest, Ytest))


# In[176]:


print("The gradient boosting regressor performs a bit better with 40% accuracy. This is a machine learning technique that combines a series of weak prediction models such as decision trees. It optimizes some cost function by always choosing the tree/function that points in the direction of interest (negative gradient). There are many other interesting models that we can employ, such as support vector machines and logistic regressors, although those most often operate on discrete variables (they are more often classifiers than numeric predictors), but with sufficient discrete parsing the interest rate problem can also be fashioned into a classificiation problem. ")

