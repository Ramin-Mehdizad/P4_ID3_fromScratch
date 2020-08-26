

"""
===============================================================================
 Created on Feb 25, 2020

 @author: Ramin Mehdizad Tekiyeh

 This code is written in Python 3.7.4 , Spyder 3.3.6
===============================================================================
"""

#==============================================================================
# deleting variables before starting main code
#==============================================================================
try:
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
except:
    print('Couldn"t erase variables from catche')


#==============================================================================
# imports
#==============================================================================
import pandas as pd
import numpy as np
import os
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn import tree
import matplotlib.pyplot as plt
import argparse

try:
    import graphviz
    print("!!! Graphviz imported successfully !!!")
except:
    print("!!! Graphviz couldn't be imported !!!")
    Var.graphviz_import_ID=0


#==============================================================================
# importing module codes
#==============================================================================
import ModFunc as Func
import ModVar as Var

    
#==============================================================================
# main program
#==============================================================================
if __name__ == '__main__':
    
    # set parameters
    Var.MaxDepth=5
    depth=0
    node_counter=0
    dot_string=''
    
    # script path
    Var.MainDir=os.path.abspath(__file__)
    Var.MainDir=Var.MainDir[0:len(Var.MainDir)-len('Main.py')]
    print('MainDir  is:  ', Var.MainDir)

    # call input data from user by means of parsing
    Func.Call_Parsed_Input()

    
    # read data and create dataframe
    df=Func.Read_Data()

    # one hot encode data
    (Var.X_trn,Var.y_trn)=Func.encod(df)
    print('X_trn   is:  \n',Var.X_trn,end='\n\n')
    print('y_trn   is:  \n',Var.y_trn,end='\n\n')
    
    # extract attribute list
    print('X_trn.shape[1]', Var.X_trn.shape[1])
    Var.attrs_list=np.arange(0,Var.X_trn.shape[1])
    print('attrs_list   is:  ',Var.attrs_list)

    # run decision tree classifier
    Trained_Tree = Func.id3(Var.X_trn, Var.y_trn, depth, max_depth=Var.MaxDepth)

    print('Trained_Tree   is:  \n\n', Trained_Tree)

    # print tree to console
    Func.Print_Tree_to_Console(Trained_Tree)
    
    # create dot_string to be used for graphviz
    dot_string, _, _ = Func.Create_dot_str(Trained_Tree, dot_string,  node_counter, depth)
    
    # tree graph is saved to a png file
    Func.render_dot_file(dot_string, 'Created_Tree')
    




