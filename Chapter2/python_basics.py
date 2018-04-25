import os
"""
curr = os.getcwd()
print(curr)
print(type(curr))

list_home = os.listdir("/home/bhavani/work")
print(list_home)
print(type(list_home))

"""
#Exception Handling
shopping_list = ["eggs","ham","milk","bacon"]

try:
    print(shopping_list[5])
except IndexError as e:
    print("Exception :"+str(e) + "has occured ")
else:
    print("No Exceptions Occured")
finally:
    print("I always execute no matter what !!")
