import pandas as pd
df=pd.read_excel(r'C:\Users\Akash\Desktop\nepal_data.xlsx')

#***************************Training Segment*****************************
ios=[]
audi=[50,66,80]
for i in range(3):
    xy_limit=int(5800*(audi[i]/100))
    a = df['risk_factor'].tolist()
    y = a[:xy_limit]
    y_bar= sum(y)/xy_limit
    l1=[]
    z1=[]
    for i in range(xy_limit):
        r1=(y[i]-y_bar)
        l1.append(r1)
        
    b = df['total_population'].tolist()
    x1 = b[:xy_limit]
    x1_bar= sum(x1)/xy_limit
    l2=[]
    p1=[]
    q1=[]
    for i in range(xy_limit):
        r2=(x1[i]-x1_bar)
        s2=(x1[i]-x1_bar)**2
        l2.append(r2)
        q1.append(s2)
        t1=l2[i]*l1[i]
        p1.append(t1)
    b1=sum(p1)/sum(q1)


    c = df['total_DAG'].tolist()
    x2 = c[:xy_limit]
    x2_bar= sum(x2)/xy_limit
    l3=[]
    p2=[]
    q2=[]
    for i in range(xy_limit):
        r3=(x2[i]-x2_bar)
        s3=(x2[i]-x2_bar)**2
        l3.append(r3)
        q2.append(s3)
        t2=l3[i]*l1[i]
        p2.append(t2)
    b2=sum(p2)/sum(q2)


    d = df['total_household'].tolist()
    x3 = d[:xy_limit]
    x3_bar= sum(x3)/xy_limit
    l4=[]
    p3=[]
    q3=[]
    for i in range(xy_limit):
        r4=(x3[i]-x3_bar)
        s4=(x3[i]-x3_bar)**2
        l4.append(r4)
        q3.append(s4)
        t3=l4[i]*l1[i]
        p3.append(t3)
    b3=sum(p3)/sum(q3)


    f = df['www'].tolist()
    x5 = f[:xy_limit]
    x5_bar= sum(x5)/xy_limit
    l6=[]
    p5=[]
    q5=[]
    for i in range(xy_limit):
        r6=(x5[i]-x5_bar)
        s6=(x5[i]-x5_bar)**2
        l6.append(r6)
        q5.append(s6)
        t5=l6[i]*l1[i]
        p5.append(t5)
    b4=(sum(p5)/sum(q5))


    g = df['DRH'].tolist()
    x6 = g[:xy_limit]
    x6_bar= sum(x6)/xy_limit
    l7=[]
    p6=[]
    q6=[]
    for i in range(xy_limit):
        r7=(x6[i]-x6_bar)
        s7=(x6[i]-x6_bar)**2
        l7.append(r7)
        q6.append(s7)
        t6=l7[i]*l1[i]
        p6.append(t6)
    b5=(sum(p6)/sum(q6))


    h = df['DDH'].tolist()
    x7 = h[:xy_limit]
    x7_bar= sum(x7)/xy_limit
    l8=[]
    p7=[]
    q7=[]
    for i in range(xy_limit):
        r8=(x7[i]-x7_bar)
        s8=(x7[i]-x7_bar)**2
        l8.append(r8)
        q7.append(s8)
        t7=l8[i]*l1[i]
        p7.append(t7)
    b6=sum(p7)/sum(q7)


    m = df['dist_gained'].tolist()
    x8 = m[:xy_limit]
    x8_bar= sum(x8)/xy_limit
    l9=[]
    p8=[]
    q8=[]
    for i in range(xy_limit):
        r9=(x8[i]-x8_bar)
        s9=(x8[i]-x8_bar)**2
        l9.append(r9)
        q8.append(s9)
        t8=l9[i]*l1[i]
        p8.append(t8)
    b7=sum(p8)/sum(q8)


    n = df['river_type'].tolist()
    x9 = n[:xy_limit]
    x9_bar= sum(x9)/xy_limit
    l10=[]
    p9=[]
    q9=[]
    for i in range(xy_limit):
        r10=(x9[i]-x9_bar)
        s10=(x9[i]-x9_bar)**2
        l10.append(r10)
        q9.append(s10)
        t9=l10[i]*l1[i]
        p9.append(t9)
    b8=sum(p9)/sum(q9)
    
#*********************************Testing Segment*************************************
    count=tp=tn=fn=fp=0
    pred=[]
    ori=[]
    b0=y_bar-(b1*x1_bar + b2*x2_bar + b3*x3_bar + b4*x5_bar + b5*x6_bar + b6*x7_bar + b7*x8_bar + b8*x9_bar)
    for i in range(xy_limit+1,5801):
        Y=(b0 + b[i]*b1 + c[i]*b2 + d[i]*b3 + f[i]*b4 + g[i]*b5 + h[i]*b6 + m[i]*b7 + n[i]*b8)
        if(abs(Y-a[i])<0.4):
            count+=1
        if(abs(Y)<0.3):
            pred.append(0)
        if(Y>0.3):
            pred.append(1) 
    acc=round(((count*100)/(5800-xy_limit)),3)
    ios.append(acc)
    for i in range(xy_limit+1,5801):
        if(a[i]>0):
            ori.append(1)
        elif(a[i]<0.3):
            #a[i]=1
           ori.append(0)
    for j in range(len(pred)):
        if(ori[j]==1 and pred[j]==1):
            tp+=1
        elif(ori[j]==0 and pred[j]==0):
            tn+=1
        elif(ori[j]==1 and pred[j]==0):
            fp+=1
        elif(ori[j]==0 and pred[j]==1):
            fn+=1
        
    print('TP=',tp,'  ','TN=',tn)
    print('FP=',fp,'  ','FN=',fn)
    EA=((tp+fp)*(tp+fn)+(fn+tn)*(fp+tn))/100

    sen=tp/(tp+fn)
    print('Sensitivity=',round(sen,2))

    spe=tn/(tn+fp)
    print('Specificity=',round(spe,2))

    prec=tp/(tp+fp)
    print('Precision=',round(prec,2))

    recall=tp/(tp+fn)
    print('Recall=',round(recall,2))

    acc2=((tp+tn)/(tp+tn+fp+fn))*100
    print('Accuracy(CM)= ',round(acc2,2))

    f1_score=(2*recall*prec)/(recall+prec)
    print('F1 Score=',round(f1_score,2))

    kap=(acc2-EA)/(100-EA)
    print('Kappa Test=',round(kap,1))
    print("\n")
