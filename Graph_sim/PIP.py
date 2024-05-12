# -*- coding:UTF-8 -*-
import numpy as np


def ATS(x,step=0):
    #step=5 #默认是0
    n=len(x)
    #print(n)
    #print(n)
    chpts=np.mat([[0,x[0]]])
    if(step==0):
        step=np.max([1,int(np.round(n/10))])
    #print(step)
    sp=0 #本来是1，但是python是从0开始的，所以改成了0
    x0=sp
    #print(np.r_[0:step+1],x[np.r_[1:(step+1)]])
    b=6*(np.sum(np.r_[0:step+1]*(x[1:(step+2)]))-x[0]*step*(step+1)/2)/(step*(step+1)*(2*step+1))
    ep=sp+step
    diff=x[ep]-x[sp]
    while(np.sign(diff)!=np.sign(b)):
        ep=ep-1
        if(ep<=0):  #本来是1，但是python是从0开始的，所以改成了0
            #print("data constant in first step")
            return
        diff=x[ep]-x[sp]
    while(diff==0):   #不希望斜率为0，宁愿往后再挪一个
        ep=ep+1
        if(ep>=(n-1)):
            #print("slope=0 and data constant after first step")
            return
        diff=x[ep]-x[sp]
    slope=(x[ep]-x[sp])/step
    cs=np.sign(slope)
    while(ep<n-1):
        #print("ep:",ep)
        spstart=sp
        while(np.sign(slope)==cs):
            ep=np.min([sp+step,n-1])
            if(sp==ep):
                break
            diff=x[ep]-x[sp]
            while(diff==0 and ep==sp+1):
                ep=ep-1
                diff=x[ep]-x[sp]
            if(diff==0):
                ep=np.min([sp+step,n-1])
                while(diff==0 and ep<n):
                    ep=ep+1
                    diff=x[ep]-x[sp]
            slope=diff/(ep-sp)
            sp=ep
        #print(spstart,sp,int(cs),np.multiply(x[spstart:sp],int(cs)),np.argmax(np.multiply(int(cs),x[spstart:sp])))
        sp=spstart+(np.argmax(np.multiply(int(cs),x[spstart:sp])))#R语言和python的a:b可能不一样
        #print("sp:",sp)
        new_chpts=[sp,x[sp]]
        chpts=np.row_stack((chpts,new_chpts))
        x0=sp
        cs=-cs
    if(chpts[chpts.shape[0]-1,0]!=n):
        new_chpts=[n-1,x[n-1]]
        chpts=np.row_stack((chpts,new_chpts))
    return(chpts)