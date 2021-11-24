# call this script from the respective folder
import os
import re
import numpy as np
import pandas as pd

for filename in os.listdir('.'):
    pre, ext = os.path.splitext(filename)
    match = re.match(r'instance(.*)', filename)
    if match:
        with open(filename, 'r') as f:
            stt = "".join(line for line in f if not line.isspace())
            split_text = re.split("<.*>+\n", stt)

            n = int(split_text[1].split('\n')[0])
            C = int(split_text[2].split('\n')[0])
            strength = split_text[3].split('\n')[0].replace(',', ".")
            strength = float(strength)


            tsk = split_text[4].split('\n')[0:-1]
            t = [int(ele.split(' ')[1]) for ele in tsk]

            prec = split_text[5].split('\n')[0:-1]
            G = [[int(x) for x in ele.split(',')] for ele in prec]
            
            pr = np.zeros((n,n))
            for i, j in G:
                pr[i-1,j-1] = 1
            pr = pr.astype(int)

            df3 = pd.DataFrame(pr)
            df2 = pd.DataFrame(t)
            df1 = pd.DataFrame([['number_of_tasks', n], ['cycle_time', C], ['order_strength', strength]])

            with pd.ExcelWriter(pre+'.xlsx') as writer:
                df1.to_excel(writer, sheet_name='data_descriptions', header=False, index=False) 
                df2.to_excel(writer, sheet_name='task_times', header=False, index=False)  
                df3.to_excel(writer, sheet_name='precedence_relations', header=False, index=False)
    
print('Done!')
# file = open("instance_n=100_1.txt", "r+")
# for line in file:
#     print(line)


        


