# # Single global optimizer cases
# func_list=["ackley","bukin","dropwave","eggholder","griewank",
#            "langermann","levy","levy13","rastrigin","schaffer2","schwefel","shubert"]

# Multiple global optimizer cases
func_list=["tray","holdertable","schaffer4"]
method_list=["autohomotopy","pinns"]
total_num = len(func_list)*len(method_list)
id = 1


with open("multiple_config.txt","w") as f:
    f.write("ArrayTaskID"+" "+"method"+" "+"func")
    for method in method_list:
        for func in func_list:
            f.write("\n"+str(id)+" "+method+" "+func)
            id+=1