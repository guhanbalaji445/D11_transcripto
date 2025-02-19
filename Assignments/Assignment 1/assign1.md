
# Python ML Basics Assignment

This assignment is designed to give you hands-on experience with some of the fundamental Python libraries used in machine learning and data science. You will work with a custom dataset (provided as a CSV file) to perform data loading, cleaning, manipulation, analysis, and visualization using NumPy, Pandas, and Matplotlib.

---

## Objective

- **Learn** how to use NumPy for efficient numerical computations.  
- **Master** data manipulation and cleaning with Pandas.  
- **Practice** data visualization techniques using Matplotlib.  
- **Explore** your custom dataset to extract meaningful insights.




## Resources for  Learning
**Refer to D11timeline.md for resources**
- **Python:**  
  - Full Python course by freecodecamp.org
  - https://www.youtube.com/watch?v=rfscVS0vtbw
  
- **NumPy:**  
  - Official NumPy documentation and beginner tutorials.  
  - https://www.youtube.com/watch?v=uRsE5WGiKWo
  
- **Pandas:**  
  - The official Pandas documentation.
  -  https://www.youtube.com/watch?v=tRKeLrwfUgU
  
- **Matplotlib:**  
  - https://www.youtube.com/watch?v=qErBw-R2Ybk
  
- **Additional Resources:**  
  - Look for video tutorials by experienced educators (for example, Corey Schafer's series on YouTube) and interactive platforms like DataCamp or Coursera for guided practical exercises.

## Pre-requisites

- Basic knowledge of Python programming.  
- Familiarity with the concepts of arrays, data frames, and plotting.  
- Installation of Python (preferably 3.x) along with the following libraries:  
  - NumPy  
  - Pandas  
  - Matplotlib  
- A working environment such as Jupyter Notebook or any Python IDE.

---

## Dataset Description

- You will work with a custom dataset file (e.g., `vgchartz-2024.csv`).  
- The dataset contains various columns including numerical, categorical, and possibly date features.  
- *Note:* Replace this description with details specific to your dataset once provided.

---

## Assignment Tasks

### 1. Data Loading and Inspection

- **Load the Dataset:**  
  - Use Pandas to load the CSV file into a DataFrame.
  - Example:
    ```
    import pandas as pd
    data = pd.read_csv('custom_dataset.csv')
    ```
- **Explore the Data:**  
  - Display the first few rows (`data.head()`) and the overall info (`data.info()`).  
  - Get summary statistics using `data.describe()`.
- **Handle Missing Values:**  
  - Identify missing data and decide whether to fill, drop, or flag these values.

---

### 2. Data Manipulation using Pandas and NumPy

- **Statistical Analysis:**  
  - Compute the mean, median, and standard deviation for total sales using both Pandas and NumPy.
    ```
    import numpy as np
    mean_val = np.mean(data['numeric_column'])
    median_val = np.median(data['numeric_column'])
    std_val = np.std(data['numeric_column'])
    ```
- **Filtering and Cleaning:**  
  - Create total_sales column by performing calculations on existing ones.
  - Filter records based on specific criteria : select rows where a total sales exceeds 13
 
- **Vectorized Operations:**  
  - Use NumPy’s vectorized functions on DataFrame columns to optimize computations.(Just here for future reference.)

---

### 3. Data Visualization with Matplotlib

- **Plotting Basics:**  
  - Create a histogram to show the distribution of a total_sales.
    ```
    import matplotlib.pyplot as plt
    plt.hist(data['numeric_column'], bins=20, color='skyblue')
    plt.title('Distribution of Numeric Column')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    ```
- **Scatter Plot:**  
  - Generate a scatter plot to analyze the relationship between two numerical variables.
    ```
    plt.scatter(data['feature1'], data['feature2'], alpha=0.7)
    plt.title('Feature1 vs Feature2')
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.show()
    ```
- **Enhancement:**  
  - Customize all plots by adding titles, axis labels, legends, and adjusting the size or color schemes.




 

## Submission Guidelines

- **Notebook/Scripts:**  
  - Organize your work in a Jupyter Notebook (.ipynb) or a Python script.  
  - Include both your code and the output (plots, tables, etc.).
- **Documentation:**  
  - Provide inline comments explaining your code logic and the steps performed.
- **Report:**  
  - Attach a brief summary (1-2 pages) discussing your findings, challenges faced, and insights derived from the data.
- **Deadline:**  
  - Submit your assignment by Tuesday, 25th feb.




---



Happy coding and good luck with your assignment!
