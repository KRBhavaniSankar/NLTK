
lambda_square = lambda n : n*n

num = 5
num_list = [1,2,3,4,5]

num_square = lambda_square(num)
list_square = [lambda_square(x) for x in num_list]
map_list_square = list(map(lambda_square, [1, 2, 3, 4, 5]))
print(num_square)
print(list_square)
print(map_list_square)


list_of_list  = [[10,20,30],[40,50],[60,70,80]]

res = [item  for each_list in list_of_list for item in each_list]
print(res)