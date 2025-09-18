# import sys

# def compare_memory_usage(data):
#     """
#     Compare memory usage between a tuple and a list containing the same data.
#     """
#     # Create tuple and list with the same data
#     tuple_data = tuple(data)
#     list_data = list(data)
    
#     # Calculate memory usage
#     tuple_size = sys.getsizeof(tuple_data)
#     list_size = sys.getsizeof(list_data)
    
#     # Calculate memory difference
#     memory_difference = list_size - tuple_size
#     percentage_difference = (memory_difference / tuple_size) * 100
    
#     # Display results
#     print(f"Data: {data}")
#     print(f"Tuple memory usage: {tuple_size} bytes")
#     print(f"List memory usage: {list_size} bytes")
#     print(f"Memory difference: {memory_difference} bytes")
#     print(f"Tuple uses {percentage_difference:.2f}% less memory")
#     print("-" * 50)
    
#     return tuple_size, list_size

# def compare_multiple_data_types():
#     """
#     Compare memory usage for different data types and sizes.
#     """
#     test_cases = [
#         [1, 2, 3, 4, 5],  # Small list of integers
#         list(range(100)),  # Medium list of integers
#         list(range(1000)),  # Large list of integers
#         ["apple", "banana", "cherry", "date"],  # Strings
#         [1.1, 2.2, 3.3, 4.4, 5.5],  # Floats
#         [1, "hello", 3.14, True],  # Mixed data types
#     ]
    
#     results = []
    
#     for i, test_data in enumerate(test_cases, 1):
#         print(f"Test Case {i}:")
#         tuple_size, list_size = compare_memory_usage(test_data)
#         results.append((tuple_size, list_size))
    
#     return results

# def detailed_memory_analysis():
#     """
#     Show detailed memory analysis for individual elements.
#     """
#     print("Detailed Memory Analysis:")
#     print("=" * 50)
    
#     # Test with individual elements
#     elements = [1, "hello", 3.14, [1, 2], (1, 2)]
    
#     for element in elements:
#         element_tuple = (element,)
#         element_list = [element]
        
#         print(f"Element: {element}")
#         print(f"  As tuple: {sys.getsizeof(element_tuple)} bytes")
#         print(f"  As list: {sys.getsizeof(element_list)} bytes")
#         print(f"  Difference: {sys.getsizeof(element_list) - sys.getsizeof(element_tuple)} bytes")
#         print()

# def memory_efficiency_analysis():
#     """
#     Analyze memory efficiency for different container sizes.
#     """
#     print("Memory Efficiency Analysis:")
#     print("=" * 50)
    
#     sizes = [10, 100, 1000, 10000]
    
#     for size in sizes:
#         data = list(range(size))
#         tuple_data = tuple(data)
#         list_data = list(data)
        
#         tuple_size = sys.getsizeof(tuple_data)
#         list_size = sys.getsizeof(list_data)
        
#         print(f"Size: {size:6d} elements")
#         print(f"  Tuple: {tuple_size:6d} bytes ({tuple_size/size:6.2f} bytes/element)")
#         print(f"  List:  {list_size:6d} bytes ({list_size/size:6.2f} bytes/element)")
#         print(f"  Savings: {list_size - tuple_size:6d} bytes")
#         print()

# if __name__ == "__main__":
#     print("Memory Usage Comparison: Tuple vs List")
#     print("=" * 50)
    
#     # Basic comparison
#     test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     compare_memory_usage(test_data)
    
#     print()
    
#     # Multiple test cases
#     compare_multiple_data_types()
    
#     print()
    
#     # Detailed analysis
#     detailed_memory_analysis()
    
#     print()
    
#     # Memory efficiency analysis
#     memory_efficiency_analysis()
    
#     print("\nConclusion:")
#     print("Tuples generally use less memory than lists because:")
#     print("1. Tuples are immutable and have a fixed size")
#     print("2. Lists need extra memory for dynamic resizing operations")
# #     print("3. Tuples have a simpler internal structure")
    
# tuple1 = (1,2,3,4,5)
# # tuple2=(4,5,6,7,8)    
# s={}
# print(type(s))
# # s={1,11,2,34,23}
# # print(s)
# # s.add(63)
# # print(s)
# # s={11,22,13,25}
# # p={98,23,13,45}
# # t={'gh','hj',66,67}
# # print(s.union(p,t))
# # print(s.intersection(p,t))
# # print(s.difference(p,t))
# # s.difference_update(p,t)
# # print(s)
# # data={1,2,3,4,5}
# # for i in range (1,6):
# #     print(data={i**2})
# # # set_int=(1,2,3,4,5)
# # # # Creating a set with integers, strings, and tuples
# # my_set = {42, "ocean", 3.14, (1, 2), True}

# # # print(my_set)
# # # Set with multiple data types
# # my_set = {42, "ocean", 3.14, (1, 2), True, 7}

# # int_set = set()


# # for item in my_set:
# #     if isinstance(item, int) and not isinstance(item, bool): 
# #         int_set.add(item)

# # print(int_set)
# # d=dict()
# # d={}
# # d[input('Enter Key')]=input('enter value')
# # print(d)
# # d.setdefault(input('enter key'):input('enter value'))
# # print(d)
# d={'Ram':30,'jai':40,'hari':25,'Shyam':45 }
# # x=d['Ram']
# # print(x)
# # print(d['Ram'])
# # print(d.get('Ram'))
# # print(d.setdefault("Ram"))
# # print(d.get("abc","key not found in dict"))
# # print(d.setdefault("Alice","key not found in the dictionary"))
# print(d.keys())
# print(d.values())
# print(d.items())
# # print()
# # print()
# l=['ram',23.67, ('shyam'.67)]
# dt={}
# # dt=dt.fromkeys(l)
# # print(dt)                                                                                                                 
d={1:['ab',34,45.67], 23:['rt','yt','lo']}
# print(d)
# d[1][1]=90
# print(d)
d['ab']['a']=90
print(d)
d['ab']['a']=4
print(d)
l=['ram',23.67,('shyam',67)]
#