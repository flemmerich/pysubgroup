'''
Created on 19.10.2017

@author: lemmerfn
'''
def check_list(my_list,threshold):
    for i in my_list:
        if i <= threshold:
            my_list.remove(i)
    return my_list

print (check_list(list(range(1,12)), 5))
            