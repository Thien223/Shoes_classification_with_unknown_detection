
import csv
if __name__=='__main__':
    	
	#n = [n for n in range(0,300,5)]
	#for i in n:
    #f = open('./accuracy_{}.csv'.format(i), 'w', encoding='utf-8')
    ans=[n for n in range(100)]
    cnt=[n for n in range(100)]
    for i in ans: ans[i] = 0
    for i in cnt: cnt[i] = 0
    all_acc = 0
    acc=[]
    count = 0
    #with open('./pred_res/prediction_res_480.csv', newline='') as f:
        # spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        # for row in spamreader:
        #    list_acc.append(row) 
        #reader = csv.reader(f)
    from csv import reader
    with open('./pred_res/prediction_res_480.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            a= row[0]
            b= row[1]
            cnt[int(float(b))]+=1
            if a==b:
                ans[int(float(b))]+=1
    for i in range(len(cnt)):
        acc.append(round(float(ans[i]/cnt[i]),2))
    print(acc)

                
    #     for row in reader:
    #         a, b=row.values()
    #         cnt[int(float(b))]+=1
    #         count += 1
    #         if a==b:
    #             #print(acc[int(float(b))])
    #             ans[int(float(b))] += 1
    #             all_acc +=1
    # #print(cnt)
    # #print(ans)
    # for i in range(len(cnt)):
    #     acc.append(round(float(ans[i]/cnt[i]),2))
        
    # print(sorted(acc))
    # print("%.4f"%(all_acc/count))
    #         # for a in row.values():
    #         #     print(a)
    # # list_acc = list(filter(None, list_acc))
    # # print(list_acc[3])
