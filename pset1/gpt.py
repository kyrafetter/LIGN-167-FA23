## LIGN 167 PSET 1
## Kyra Fetter, A16819058
## Lina Battikha, A16852105
## Oishani Bandopadhyay, A1600827

#  GPT 3.5 

# 1

# 2 

# 3

# 4

# 5

# 6

# 7 

# 8 

# 9 

# 10

# 11

# GPT 4

# 1

# 2 

# 3

# 4

# 5

def dL_db_gpt_4(x_vals, y_vals, a, b):
    """
    Compute the partial derivative of L with respect to b.
    
    Parameters:
    - x_vals: 1-D numpy array of x values.
    - y_vals: 1-D numpy array of y values.
    - a, b: Parameters of the function f.
    
    Returns:
    - dLdb: Partial derivative of L with respect to b.
    """
    n = len(x_vals)
    # Compute the derivative for each pair (x_i, y_i)
    df_db = 2 * (y_vals - a * x_vals - b) * (-1)
    # Sum all the derivatives and divide by n
    dLdb = np.sum(df_db) / n
    return dLdb

# Example usage:
# x_vals = np.array([1, 2, 3, 4])
# y_vals = np.array([2, 4, 5, 4])
# a = 1
# b = 2
# print(dL_db(x_vals, y_vals, a, b))


# 6

# 7

# 8 

# 9 

# 10 

# 11
