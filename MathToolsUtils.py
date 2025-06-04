"""
MathToolsUtils.py

This file contain various functions relating to
linear algebra and differential equations:
- Determinant calculator
- ODE solution approximation
- Linear system solver

Author: James Milgram
"""
from sympy import symbols, sympify
import matplotlib.pyplot as plt
import numpy as np
from sympy import solve, Eq, Matrix, symbols, det, exp, I

def function_cleanse(f):
    """
    RETURNS THE ODE
    JUST WITH X TERMS
    f : Function in terms of y and x (string)
    RETURNS A SYMPIFY FORMULA 
    """
    x = symbols("x")
    y = symbols("y")
    if "=" in f:
        if "dy/dx" not in f:
            raise SyntaxError("Incorrect function syntax!")
        else:
            pass
    f = sympify([x for x in f.split("=") if "d" not in x][0]) 
    return f
    
def forward_eulers_method(f, d, start_y, start_x, end_x):
    """
    forward_eulers_method : Forward Euler's Method
    APPROXIMATES THE VALUE OF A
    FUNCTION (SOLUTION TO AN ODE) AT A POINT
    REQUIRES FROM SYMPY IMPORT SYMBOLS, SYMPIFY
    f : ODE in terms of y and x (string)
    d : Step size (scalar)
    start_y : Starting y-value (scalar)
    start_x : Starting x-value (scalar)
    end_x : Desired x-value (scalar)
    RETURNS A FLOAT
    """
    if not ((end_x-start_x)/d).is_integer():
        raise ValueError("The step size is incompatiable! Try again with a different step size (d)!")
    iters = int((end_x-start_x)/d)
    f = function_cleanse(f)
    x = symbols("x")
    y = symbols("y")
    x_val = 0
    y_n = 0
    history_dict = {}
    for i in range(iters):
        if i == 0:
            y_n = start_y
            history_dict[i] = y_n
            x_val = start_x
            dfdx = f.subs(x, x_val).subs(y,y_n).evalf()
            y_n1 = y_n + (d*dfdx)
            x_val += d
            y_n = y_n1
            history_dict[i+1] = y_n
        else:
            dfdx = f.subs(x, x_val).subs(y,y_n).evalf()
            y_n1 = y_n + (d*dfdx)
            x_val += d
            y_n = y_n1
            history_dict[i+1] = y_n
    
    print(f"Forward Euler's Method with the function: {f}\n"
          f"Initial x-value: {start_x}\n"
          f"Initial y-value: {start_y}\n"
          f"Step size: {d}\n"
          f"The value of y at {end_x} is approximately {y_n:.10f}")

    return y_n, history_dict

def rk2(f, d, start_y, start_x, end_x):
    """
    rk2 : Runge-Kutta 2nd Order Method
    APPROXIMATES THE VALUE OF A
    FUNCTION (SOLUTION TO AN ODE) AT A POINT
    REQUIRES FROM SYMPY IMPORT SYMBOLS, SYMPIFY
    f : ODE in terms of y and x (string)
    d : Step size (scalar)
    start_y : Starting y-value (scalar)
    start_x : Starting x-value (scalar)
    end_x : Desired x-value (scalar)
    RETURNS A FLOAT
    """
    if not ((end_x-start_x)/d).is_integer():
        raise ValueError("The step size is incompatiable! Try again with a different step size (d)!")
    iters = int((end_x-start_x) / d)
    f = function_cleanse(f)
    x = symbols("x")
    y = symbols("y")
    
    for i in range(iters):
        if i == 0:
            y_n = start_y
            x_val = start_x
            k1 = f.subs(x, x_val).subs(y, y_n).evalf()
            k2 = f.subs(y, y_n + (d*k1)).subs(x, x_val + d).evalf()
            y_n1 = y_n + (.5 * d * (k1 + k2))
            x_val += d
            y_n = y_n1
        else:
            k1 = f.subs(x, x_val).subs(y, y_n).evalf()
            k2 = f.subs(y, y_n + (d*k1)).subs(x, x_val + d).evalf()
            y_n1 = y_n + (.5 * d * (k1 + k2))
            x_val += d
            y_n = y_n1

    print(f"Runge-Kutta 2nd Order Method with the function: {f}\n"
          f"Initial x-value: {start_x}\n"
          f"Initial y-value: {start_y}\n"
          f"Step size: {d}\n"
          f"The value of y at {end_x} is approximately {y_n:.10f}")
    
    return y_n

def determinant(array):
    """
    FINDS THE DETERMINANT OF
    A 2D ARRAY
    REQUIRES NUMPY AS NP
    USES 1ST ROW COFACTOR EXPANSION
    array : a 2D array (matrix)
    RETURNS A FLOAT
    """
    r,c = array.shape
    if not r == c:
        raise ValueError("Not a square matrix! Determinant is not defined!") 
    detr = 0
    if r >= 3:
        for j in range(c):
            if j % 2 == 0:
                a = array[0][j]
            else:
                a = -1 * array[0][j]
            sm = np.delete(array, 0, axis=0)
            sm = np.delete(sm, j, axis=1)
            detr += determinant(sm) * a
            
        
    else:
        detr += (array[0][0] * array[1][1] - array[0][1] * array[1][0])

    return detr

def determinant_any_row(array, row):
    """
    FINDS THE DETERMINANT OF
    A 2D ARRAY
    USES ith ROW COFACTOR EXPANSION
    REQUIRES NUMPY AS NP
    array : a 2D array (matrix)
    row* : desired row (scalar)
    RETURNS A FLOAT
    
    *Uses Linear Algebra indexing
    for rows (starting from 1)
    
    """
    r,c = array.shape
    if not r == c:
        raise ValueError("Not a square matrix! Determinant is not defined!")
    if not row in range(1, r+1):
        raise ValueError("Row value is does not reflect the number of rows in the matrix!")
    detr = 0
    if r >= 3:
        for j in range(c):
            if (j + (row-1)) % 2 == 0:
                a = array[row-1][j]
            else:
                a = -1 * array[row-1][j]
            sm = np.delete(array, row-1, axis=0)
            sm = np.delete(sm, j, axis=1)
            detr += determinant(sm) * a
            
        
    else:
        detr += (array[0][0] * array[1][1] - array[0][1] * array[1][0])

    return detr

def linear_system_solver_2x2(A):
    """
    linear_system_solver_2x2 : Linear System Solver
    FINDS THE VECTOR X THAT SOLVES
    THE GENERAL EQUATION AX = X'
    REQUIRES NUMPY AS NP,
    FROM SYMPY IMPORT SOLVE, EQ,
    MATRIX, SYMBOLS, DET, EXP I
    A : 2x2 matrix (array)
    RETURNS A DICTIONARY
    DESCRIBING X1 and X2
    WHERE XH = X1 + X2,
    PRINTS XH
    """
    r, c = A.shape
    if not r == c == 2:
        raise ValueError("Input is not a 2 x 2 matrix!")
    
    λ = symbols("λ")
    t = symbols("t")
    c1 = symbols("c1")
    c2 = symbols("c2")

    
    A = Matrix(A)
    I_matrix = Matrix.eye(r)
    A_minus_lambda_I = A - (λ * I_matrix)
    detr = A_minus_lambda_I.det()
    eq = Eq(detr, 0)
    λ = solve(eq)
    
    if len(λ) == 2:
        vectors = np.empty((0,2), dtype=float)
        for val in λ:
            v1 = symbols("v1")
            v2 = symbols("v2")
    
            m = A - (val * I_matrix)
            eq = ((m[0,0] * v1) + (m[0,1] * v2)).subs(v1,1)
            v2 = solve(Eq(eq, 0))[0]
            v = np.array([1, v2])
            vectors = np.vstack([vectors, v])
    
        x_h = {"x1": list(c1 * vectors[0] * exp(λ[0] * t)) , "x2": list(c2 * vectors[1] * exp(λ[1] * t))}
        print("The homogeneous solution to the equation Ax = x' is:")
        print(f"x_h = c1 * {vectors[0]} * {exp(λ[0] * t)} + c2 * {vectors[1]} * {exp(λ[1] * t)}")
        
        return x_h
        
    else:
        vectors = np.empty((0,2), dtype=float)
        val = λ[0]
        v1 = symbols("v1")
        v2 = symbols("v2")
        w1 = symbols("w1")
        w2 = symbols("w2")
    
        m = A - (val * I_matrix)
        eq = ((m[0,0] * v1) + (m[0,1] * v2)).subs(v1,1)
        v2 = solve(Eq(eq, 0))[0]
        v = np.array([1, v2])
        vectors = np.vstack([vectors, v])

        w = solve(Eq(((m[0,0] * w1) + (m[0,1] * w2)).subs(w1,1),v[0]))[0]
        w = np.array([1, w])
        vectors = np.vstack([vectors, w])

        x_h = {"x1": list(c1 * vectors[0] * exp(λ[0] * t)) , "x2": list(c2 * (t * vectors[0] * exp(λ[0] * t) + vectors[1] * exp(λ[0] * t)))}
        print("The homogeneous solution to the equation Ax = x' is:")
        print(f"x_h = (c1 * {vectors[0]} * {exp(λ[0] * t)}) + c2(t * {vectors[0]} * {exp(λ[0] * t)} + {vectors[1]} * {exp(λ[0] * t)})")

        return x_h
