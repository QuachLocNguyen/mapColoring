# Tài liệu tham khảo
# A. Artasanchez, P. Joshi, 
# Artificial Intelligence with Python, 
# 2nd Edition, Packt, 2020
# Trang 234
import streamlit as st
import cv2
import numpy as np

from simpleai.search import CspProblem, backtrack
# Define the function that imposes the constraint 
# that neighbors should be different

def constraint_func(names, values):
    return values[0] != values[1]

if __name__=='__main__':
    # Specify the variables
    names = ('Mark', 'Julia', 'Steve', 'Amanda', 'Brian',
    'Joanne', 'Derek', 'Allan', 'Michelle', 'Kelly', 'Chris')

    names_point = [(126, 70),(430,92),(120,300),(357,295),(680,80),
                   (540,480),(580,280),(45,500),(280,500),(735,275),(750,480)]

    color_available = ['red', 'green', 'blue', 'gray']

    # Define the possible colors
    colors = dict((name, ['red', 'green', 'blue', 'gray']) for name in names)

    # Define the constraints
    constraints = [
        (('Mark', 'Julia'), constraint_func),
        (('Mark', 'Steve'), constraint_func),
        (('Julia', 'Steve'), constraint_func),
        (('Julia', 'Amanda'), constraint_func),
        (('Julia', 'Derek'), constraint_func),
        (('Julia', 'Brian'), constraint_func),
        (('Steve', 'Amanda'), constraint_func),
        (('Steve', 'Allan'), constraint_func),
        (('Steve', 'Michelle'), constraint_func),
        (('Amanda', 'Michelle'), constraint_func),
        (('Amanda', 'Joanne'), constraint_func),
        (('Amanda', 'Derek'), constraint_func),
        (('Brian', 'Derek'), constraint_func),
        (('Brian', 'Kelly'), constraint_func),
        (('Joanne', 'Michelle'), constraint_func),
        (('Joanne', 'Amanda'), constraint_func),
        (('Joanne', 'Derek'), constraint_func),
        (('Joanne', 'Kelly'), constraint_func),
        (('Joanne', 'Chris'), constraint_func),
        (('Derek', 'Kelly'), constraint_func),
        (('Derek', 'Chris'), constraint_func),
        (('Chris', 'Kelly'), constraint_func),
    ]

    # Solve the problem
    problem = CspProblem(names, colors, constraints)
    # Print the solution
    result = backtrack(problem) 
    print('\nColor mapping:\n') 
    for k, v in result.items():
        print(k, '==>', v)

    image = cv2.imread('regions.jpg', cv2.IMREAD_GRAYSCALE)
    M, N = image.shape

    # Phân ngưỡng để ảnh chỉ còn 2 giá trị là 0 và 255
    val, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    image_s = image.copy()
    image_s = cv2.cvtColor(image_s, cv2.COLOR_GRAY2BGR)

    mau_xam = [(10,10,10), (50,50,50), (100,100,100), (150,150,150)]
    mau_mau = [(0,0,255), (0,255,0), (255,0,0), (128,128,128)]
 
    mask = np.zeros((M + 2, N + 2), np.uint8)

    for k, v in result.items():
        vi_tri_point = names.index(k)
        point = names_point[vi_tri_point]
        vi_tri_mau = color_available.index(v)
        mau = mau_xam[vi_tri_mau]
        cv2.floodFill(image, mask, point, mau)

    for x in range(0, M):
        for y in range(0, N):
            r = image[x,y]
            if r > 0:
                r = (r,r,r)
                vi_tri = mau_xam.index(r)
                mau = mau_mau[vi_tri]
                image_s[x,y,:] = mau
    cv2.imshow('Image', image_s)
    st.image(image_s, channels="RGB")
    cv2.waitKey()
