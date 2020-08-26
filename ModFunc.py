
"""
===============================================================================
 Created on Feb 25, 2020

 @author: Ramin Mehdizad Tekiyeh

 This code is written in Python 3.7.4 , Spyder 3.3.6
===============================================================================
"""


#==============================================================================
# importing standard classes
#==============================================================================
import logging
import numpy as np
import pandas as pd
import os
import argparse
import sys

try:
    import graphviz
    print("!!! Graphviz imported successfully !!!")
except:
    print("!!! Graphviz couldn't be imported !!!")
    Var.graphviz_import_ID=0



#==============================================================================
# importing module codes
#==============================================================================
import ModVar as Var
import Main as Mn


#==============================================================================
# this function asks user input data
#==============================================================================  
def Input_Data_Message():
    
    print('')
    print('|===========================================================')
    print('|  ==> To run the code with default values, just press Enter')
    print('|  ==> Otherwise:')
    print('|  ==> Enter the parameters as following format:')
    print('|')
    print('| -l 0 -p 0')
    print('|')
    print('|  ==> To get help, type "-h" and press Enter')
    print('|  ==> To exit program, type "Q" and press Enter')
    print('|===========================================================')
    
    Var.str_input=input('  Enter parameters: ').strip()


#==============================================================================
# this function asks user input data
#==============================================================================
def Call_Parsed_Input():
    
    # create parse class
    parser1 = argparse.ArgumentParser(add_help=True,prog='Decision Tree ID3 Scratch',
             description='* This program is based on ID3 scratch code')
    
    # set program version
    parser1.add_argument('-v','--version',action='version',
                        version='%(prog)s 1.0')

    # whether to create log file or not
    parser1.add_argument('-l', '--log', action='store',
                         default='1', dest='logFile', choices=['0', '1'],
                         help='0: Dont create logfile     1: create logfile')
    
    # whether to print calculations
    parser1.add_argument('-p', '--PrintCalculations', action='store', 
                         default='1',  dest='PrintCalculations', choices=['0', '1'],
                         help='0: Dont Print Calculations     1: Print Calculations')
    
    
    # indicates when to exit while loop
    entry=False
    while entry==False:
        # initialize
        ParsErr=0
        FileErr=0
        makedirErr=0
        
        # --------------in this section we try to parse successfully-----------
        # function to call input data from command line    
        Input_Data_Message()
        
        # user wanted to continue with default values
        if Var.str_input=='':
            Var.args=parser1.parse_args()
            # exit while loop
            entry=True
        elif Var.str_input.upper()=='Q':
            # exit script
            sys.exit()
        else:
            entry=True
            ParsErr=0
            try:
                args=parser1.parse_args(Var.str_input.split(' '))
            except:
                entry=False
                ParsErr=1
                
        #----------------------------------------------------------------------
             
        if Var.args.PrintCalculations=='1':
            Var.FlagPrintSplitData=True
                            
        #----------------------------------------------------------------------


#==============================================================================
# this function asks user input data
#==============================================================================  
def Read_Data():

    df=pd.read_csv('dataset.csv')
    
    print(df,end='\n\n')
    
    return df


#==============================================================================
# this function encods data
#==============================================================================
def encod(df_data):
    
    # a list of attr names
    Var.column_names=list(df_data.columns)
    num_columns=len(Var.column_names)
    if Var.FlagPrintSplitData: print('num of columns is:',num_columns)
    
    # number of data record
    num_data=len(df_data[Var.column_names[0]])
    if Var.FlagPrintSplitData: print('num of data is:',num_data)
    
    # extract list of each column unique values
    Var.column_unique_values=list()
    for i in range(num_columns):
        vals=list()
        for j in range(num_data):
            if df_data.iloc[j][i] not in vals:
                vals.append(df_data.iloc[j][i])
        Var.column_unique_values.append(vals)
        
    # encode values of each column
    encoded_data=list()
    for i in range(num_columns): 
        encoded_column_vals=list()
        for j in range(num_data):
            for n in range(len(Var.column_unique_values[i])):
                if df_data.iloc[j][i]==Var.column_unique_values[i][n]:
                    encoded_column_vals.append(n)
        encoded_data.append(encoded_column_vals)
    encoded_data=np.array(encoded_data).T
    
    X_trn=encoded_data[:,0:-1]
    y_trn=encoded_data[:,-1]
    
    return(X_trn,y_trn)
 

#==============================================================================
# this function encods data
#==============================================================================
def Labels_Unique_List(y):
    Labels_list=list()
    for _,j in enumerate(y):
        if j not in Labels_list:
            Labels_list.append(j)
    return Labels_list


#==============================================================================
# this function encods data
#==============================================================================
def y_Majority(y):
    if Var.FlagPrintSplitData: print('entered y_Majority func')
    Labels_list=Labels_Unique_List(y)
    if Var.FlagPrintSplitData: print('Labels_list is: ', Labels_list)
    
    if len(Labels_list)==1:
        lbl=Var.column_unique_values[-1][int(y[0])]
    elif len(Labels_list)==2:
        # we return majority label
        L1=int(Labels_list[0])
        L2=int(Labels_list[1])
        if Var.FlagPrintSplitData: 
            print('L1 is:  ',L1) 
            print('L2 is:  ',L2)
        n1=0
        n2=0
        if Var.FlagPrintSplitData: print('len(y) is:  ', len(y))
        for i in range(len(y)):
            if y[i]==L1: n1+=1
            if y[i]==L2: n2+=1
        if Var.FlagPrintSplitData: 
            print('n1 is:  ',n1) 
            print('n2 is:  ',n2)
        if n1 > n2: 
            lbl=Var.column_unique_values[-1][int(L1)]
        elif  n1==n2 :
            lbl=Var.column_unique_values[-1][int(L1)]
        else:
            lbl=Var.column_unique_values[-1][int(L2)]
    
    if Var.FlagPrintSplitData: print('y majority is:  ',str(lbl))        
    return lbl


#==============================================================================
# this function encods data
#==============================================================================
def partition(x):
    # here we extract unique values
    unique_vals=list()
    dict_partitioned=dict()
    for _ ,xx in enumerate(x):
        if xx not in unique_vals:
            unique_vals.append(xx)
            
    # here we fill dict for unique_vals
    for i in range(len(unique_vals)):
        indices_list=list()
        for j, xx in enumerate(x):
            if xx==unique_vals[i]: indices_list.append(j)
        dict_partitioned[unique_vals[i]]= indices_list 

    return dict_partitioned


#==============================================================================
# this function encods data
#==============================================================================
def entropy(x):
    x_partitioned=partition(x)
    if len(x_partitioned)==1:
        if Var.FlagPrintSplitData: print('!!!!!!--- Ent=0 ---!!!!!!')
        return 0
    elif len(x_partitioned)==3:
        if Var.FlagPrintSplitData: 
            print('!!! Be careful: not binary partitioning !!!')
        return
    elif len(x_partitioned)==2:
        Ent=0
        keys=list(x_partitioned.keys())
        n_Tot=len(x)
        n=[len(x_partitioned[keys[0]]),len(x_partitioned[keys[1]])]
        for i in range(2):
            Pi=n[i]/n_Tot
            Ent=Ent-Pi*np.log2(Pi)
        return Ent    


#==============================================================================
# this function encods data
#==============================================================================
def Info_Gain(attr_no,X_trn,y_trn):
    
    # global column_values, column_names, column_unique_values
    # global num_columns, ID_list
    
    if Var.FlagPrintSplitData: 
        print('\n\n--------  New Info_Gain Calc --------')
        print('current attr_no for calc of Info_Gain is :  ',attr_no)
    
    Ent_S=entropy(y_trn)
    if Var.FlagPrintSplitData: print('Ent_S is :  ',Ent_S)
    
    # initialize info_gain
    info_gain=Ent_S

    x_partitioned=partition(X_trn[:,attr_no])
    if Var.FlagPrintSplitData: print('x_partitioned is: ', x_partitioned)
    
    if len(x_partitioned)==1:
        # it means that by use of this attr all data falls in the same group
        # so we get no info_gain
        # return info_gain
        return 0
    else:
        # total num of data
        n_Tot=len(X_trn[:,0])
        
        for _ , indice in x_partitioned.items():
            y_if_Sv=[y_trn[i] for i in indice]
            if Var.FlagPrintSplitData: print('y_if_Sv  ',y_if_Sv)
            info_gain=info_gain-(len(y_if_Sv)/n_Tot)*entropy(y_if_Sv)
            
        if Var.FlagPrintSplitData: 
            print('info_gain  of this attr is:  ',info_gain)
            print('-------------------------------------')

        return info_gain        


#==============================================================================
# this function encods data
#==============================================================================
def id3(X_trn, y_trn, depth, max_depth=10):
    
    # depth incremented
    depth=depth+1
    if Var.FlagPrintSplitData: 
        print('\n\n<<<<=======id3 depth of: '+str(depth)+' staretd=======>>>>')
    
    # extract class labels
    Labels_list=Labels_Unique_List(y_trn)
    if Var.FlagPrintSplitData: 
        print('Labels_list   is:  ',Labels_list)
        print('No of Data in this id3 depth is:   ', len(y_trn))

    #-------------------- Base Case: ending conditions-------------------------
    if depth > max_depth:
        if Var.FlagPrintSplitData: print('max n_iter reached')
        return y_Majority(y_trn)
    else:
        if len(Var.attrs_list) ==0:
            if Var.FlagPrintSplitData: print('\n All attributes are used up')
            return y_Majority(y_trn)
        else: 
            # I put the following 'if' so that it wont continue into info-gain
            # function and avoid any possible errors
            if len(Labels_list)==1:
                return y_Majority(y_trn)
            else:
                
                #-------------------- Recursive Case: -------------------------
                    
                # choose best attr by calc of info-gains-----------------------  
                # it means only one attribute is left so we choose it
                Info_Gain_list=list()
                for i in range(len(Var.attrs_list)):
                    # attr_no=attrs_list[i]
                    info_gain=Info_Gain(Var.attrs_list[i],X_trn,y_trn)
                    Info_Gain_list.append(info_gain)
                if Var.FlagPrintSplitData: 
                    print('\n Info_Gain_list of current depth is:  \n',
                                              Info_Gain_list,end='\n\n')
                
                # select best info gain and best attribute
                max_info_gain=0
                best_attr_ID=0
                # check if only one attr is left
                if len(Var.attrs_list)==1:
                    max_info_gain=Info_Gain_list[0]
                    best_attr_ID=0
                else:
                    for i in range(len(Var.attrs_list)):
                        if Info_Gain_list[i] > max_info_gain:
                            max_info_gain=Info_Gain_list[i]
                            best_attr_ID=Var.attrs_list[i]
                if Var.FlagPrintSplitData:             
                    print('max_info_gain is:   ',max_info_gain,end='\n\n')            
                    print('best attr ID is:   ',best_attr_ID,end='\n\n')
                #--------------------------------------------------------------
                    
                
                # the following means we wont get any gain if we continue
                # the process
                if max_info_gain==0:
                    return y_Majority(y_trn)
            
                # calc tuple for tree dict
                tuples_list=list()
                for _, v in enumerate(Var.column_unique_values[best_attr_ID]):
                    tuples_list.append((Var.column_names[best_attr_ID],v))
                
                # print tuples created for tree dict
                print('tuples_list is:   ',tuples_list,end='\n\n')
                
                # now we split data based on selected attribute to continue 
                # recursion process
                x_partitioned=partition(X_trn[:,best_attr_ID])
                if Var.FlagPrintSplitData: 
                    print('x_partitioned is: ', x_partitioned)
                ID_Split_list=list()
                curr_num_data=X_trn.shape[0]
                
                if Var.FlagPrintSplitData: 
                    print('curr_num_data is:',curr_num_data,end='\n\n')
                    
                for i in range(len(Var.column_unique_values[best_attr_ID])):
                    curr_val_ID=list()
                    
                    for j in range(curr_num_data):
                        if X_trn[j,best_attr_ID]==i:
                            curr_val_ID.append(j)
                    ID_Split_list.append(curr_val_ID)
                    
                if Var.FlagPrintSplitData: 
                    print('ID_Split_list is:',ID_Split_list,end='\n\n')
                
                # I double check it here for any unforseen problems
                if len(ID_Split_list)==0:
                    return y_Majority(y_trn)
            
                # extract split x and y        
                X_trn_split_list=list()
                y_trn_split_list=list()
                for _, v in enumerate(ID_Split_list):
                    xxx=X_trn[v,:]
                    yyy=y_trn[v]
                    X_trn_split_list.append(xxx)
                    y_trn_split_list.append(yyy)
                    
                if Var.FlagPrintSplitData: print('split x in this depth is: \n',
                                X_trn_split_list,end='\n\n')
                if Var.FlagPrintSplitData: print('split y in this depth is: \n',
                                y_trn_split_list,end='\n\n')
               
                # eliminate the calculated attribute
                index=np.where(Var.attrs_list==best_attr_ID)[0]
                Var.attrs_list=np.delete(Var.attrs_list, index)

                # initialize tree
                Trained_Tree=dict()
                
                for i in range(len(ID_Split_list)):
                    Trained_Tree[tuples_list[i]]=id3(X_trn_split_list[i],
                           y_trn_split_list[i], depth, max_depth=Var.MaxDepth)
                
                print('Trained_Tree is:  \n', Trained_Tree)
                
                return Trained_Tree


#==============================================================================
# this function encods data
#==============================================================================    
def Print_Tree_to_Console(tree, depth=0):

    if depth==0:
        print('\n\n&&&&&&&&&&&&&&&& === TREE === &&&&&&&&&&&&&&&&&&')
        for _ in range(4):
            print('              ||               ||')

        print('         \\\        //     \\\        //')
        print('          \\\      //       \\\      //')
        print('           \\\    //         \\\    //')
        print('            \\\  //           \\\  //')
        print('             \\\//             \\\//')
        print('              \\/               \\/')
        print('\nTREE')

    for index, split_criterion in enumerate(tree):
        sub_trees=tree[split_criterion]

        # Print the current node: split criterion
        print('|\t'*depth,end='')
        print('+---[@@@@: {0} = {1} ]'.format(split_criterion[0], 
                                split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            Print_Tree_to_Console(sub_trees,depth+1)
        else:
            print('|\t'*(depth+1),end='')
            print('+---[LABEL ===> {0}]'.format(sub_trees))   



#==============================================================================
# this function encods data
#==============================================================================    
def Create_dot_str(tree, dot_string='', node_counter=0, depth=0):

    print('\n=========== id3 recursion  ===========')
    print('entered dot_string is:  \n', dot_string)
    print('entered node_counter is:  ', node_counter)

    node_counter += 1       # Running index of node ids across recursion
    cur_node_id = node_counter  # Node id of this node
    
    print('node_counter   is:  ', node_counter)
    print('cur_node_id   is:  ', cur_node_id)
    

    # when depth=0, the first line of dot_string is added here
    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for key, value in tree.items():
        
        sub_trees = value
        attribute_name = key[0]
        attribute_value = key[1]

        print('for this for loop we have: >>>')
        print('sub_trees   is:  ', sub_trees)
        print('attribute_name   is:  ', attribute_name)
        print('attribute_value   is:  ', attribute_value)
        

        if type(sub_trees) is dict:
            
            print('type dict')

            # data before recursion
            print('cur_node_id   is:  ', cur_node_id)      
            new_node_id=node_counter+1
            print('node_counter   is:  ', node_counter)
            print('new_node_id   is:  ', new_node_id)


            print('dot_string BEFORE RECURSION  is:  \n', dot_string)
            
            dot_string, cur_node_id, node_counter  = Create_dot_str(\
                sub_trees, dot_string, node_counter, depth+1 )
            
            # create old node
            dot_string += 'node{0} [label="{1}"];\n'.format(cur_node_id, attribute_name)
                
            # create new node
            dot_string += 'node{0} [label="{1}"];\n'.format(new_node_id, '')
            
            # create new connection
            dot_string += 'node{0} -> node{1} [label="{2}"];\n'.format(cur_node_id, 
                                        node_counter+1, attribute_value)

        else:
            
            print('if else')

  
            # increase counter to craete new label node
            node_counter+=1
            new_node_id=node_counter
            
            old_node_id=cur_node_id
            
            print('if else with: cur_node_id={0}, new_node_id={1}'.format(cur_node_id, new_node_id))
            
            # create current node
            dot_string += 'node{0} [label="{1}"];\n'.format(cur_node_id, attribute_name)
            
            # create new node
            dot_string += 'node{0} [label="{1}"];\n'.format(new_node_id, sub_trees)
            
            # create connection
            dot_string += 'node{0} -> node{1} [label="{2}"];\n'.format(cur_node_id,
                                            new_node_id, attribute_value)
            
            print('dot_string   is:  \n', dot_string)

    
    # when depth=0, the last line of dot_string is added here
    if depth == 0:
        dot_string += '}\n'
    
    return dot_string, cur_node_id, node_counter



#==============================================================================
# Uses GraphViz to render a dot file
#==============================================================================
def render_dot_file(dot_string, save_file, image_format='png'):

    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # # Set path to your GraphViz executable here
    try:
        graph = graphviz.Source(dot_string)
        graph.format = image_format
        graph.render(save_file, view=True) 
    except:
        print('Error using graphviz') 









