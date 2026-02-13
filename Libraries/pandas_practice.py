import pandas as pd
# import numpy as np

# 1. creating sample dataset
data={"Name":["Harsha","Rahul","Sai"],
      "Age":[21,25,23],
      "Department":["IT","HR","IT"],
      "Salary":[40000,50000,45000],
      "Experience":[1,2,1]}
df=pd.DataFrame(data)
print(df)

# 2. basic information
# this gives data type info
print(df.info()) 

# this gives statistics
print(df.describe())

# 3. handling missing values

df["Age"].fillna(df["Age"].mean())
df["Salary"].fillna(df["Salary"].median())
# --if we are dealing in df's
df.isnull().sum()   #this gives overall null values in each col
df.fillna(0) # this will fill null with 0's
df.dropna() # this will drop rows with null values


# 4. filtering data
# this gives only employes with particular salary
salary_filterd=df[df["Salary"]==40000]

#5. feature engineering
# it creates seperate column in df 
df["Salary_per_Year_of_Exp"] = df["Salary"] / df["Experience"]
print("After Feature Engineering:")
print(df)

# 6. groupBy operations

# when we use groupby we will always use categorical col with numerical col
grouped = df.groupby("Department")["Salary"].mean()
print("Average Salary by Department:")
print(grouped)

# 7. sorting

sorted_df = df.sort_values(by="Salary", ascending=False)
print("Sorted by Salary (Descending):")
print(sorted_df)

# 8.data loading in pd

df=pd.read_csv("data.csv")
df=pd.read_excel("data.xlsx")
df=pd.read_json("data.json")

9.# exploaring data
df.head()      # First 5 rows
df.tail()      # Last 5 rows

df.shape       # (rows, columns)
df.columns     # Column names

#10. removing duplicates
df.drop_duplicates()

# 11.change the datatypes
df["Age"]=df["Age"].astype(int)

# 13.using groupby with aggregation
df.groupby("Department").agg({
    "Salary": ["mean", "max", "min"],
    "Age": "mean"
})

#14.working with data and time
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year


#*** Advance pandas

# apply() and lambda
df["bonus"]=df["Salary"].apply(lambda x: x*0.10) # needed for custom transformations

# vectorization
# Pd performs elementwise addition
df["Total"] = df["Salary"] + df["Bonus"]

# multiIndex
df.set_index(["Department", "Name"])

# merge
pd.merge(df1, df2, on="ID", how="inner")# it keeps only id's which are same

# concatenation
pd.concat([df1, df2], axis=0,ignore_index=True)

# pivot table
pd.pivot_table(df, values="Salary", index="Department", aggfunc="mean")

# rolling mean
df["Salary"].rolling(window=3).mean()

# memory optimization
df["Department"] = df["Department"].astype("category")  # for memory saving

#quering
df.query("Salary > 50000")
# Same as:
df[df["Salary"] > 50000]

#chaining
result = (
    df.dropna()
      .query("Salary > 40000")
      .groupby("Department")
      .mean()
)