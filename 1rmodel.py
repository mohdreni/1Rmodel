import pandas as pd
import numpy as np
import copy
import os


def cat_attribute_accuracy(attrname,attrvals,outcome_values):
    # first parameter is the current attribute 2nd is the outcome value
    attribute_values=np.array(attrvals).astype(str)
    out_values= np.array(outcome_values)
    primery_key=attribute_values + out_values

    # Getting the count of primary key
    attribute_values_cnt = []
    counted_attr=[]
    for it in primery_key:
        cnt = 0
        if it not in counted_attr:
            for it_cnt in primery_key:
                if it == it_cnt:
                    cnt = cnt+1
            counted_attr.append(it)
            attribute_values_cnt.append(cnt)
            #print ( 'Count of %s is %d'%(it,cnt))

    min_cnt_attr_lst=counted_attr[:]

    total_error = 0
    origin_attr_val=[]
    attr_element_prediction={}
    #Traverse thru each element to find min and max occurance primary key count .This will help cal calc error.
    for cnt_ittr in attribute_values:
        #exit when it finish traversing all distict values of element
        if len(min_cnt_attr_lst) == 0:
            break
        #check to avoid duplicate traversing
        if cnt_ittr not in origin_attr_val:
            origin_attr_val.append(cnt_ittr)
            rawattrsize=len(cnt_ittr)

            cnt_index=0
            min_val=-1
            max_val=-1
            min_attr_nme = ''
            max_attr_nme = ''

            #To handle only one occurance of counted attribute
            cnt_attr_occurance= 0
            for min_max_cnt_attr in counted_attr:
                if min_max_cnt_attr in min_cnt_attr_lst:
                    if min_max_cnt_attr[0:rawattrsize] == cnt_ittr:
                        cnt_attr_occurance = cnt_attr_occurance+1
                        #print('The values %s is %d' %(cnt_ittr,attribute_values_cnt[cnt_index]))
                        current_val=attribute_values_cnt[cnt_index]
                        current_attr_nme=min_max_cnt_attr
                        #finiding minimum value which will use as error
                        if min_val == -1:
                            min_val = current_val
                            min_attr_nme=min_max_cnt_attr
                        if min_val <= current_val:
                            min_val = min_val
                            min_attr_nme=min_attr_nme
                        else:
                            min_val = current_val
                            min_attr_nme=min_max_cnt_attr
                        #finiding max value which will use as predictor result of each element
                        if max_val == -1:
                            max_val=current_val
                            max_attr_nme = min_max_cnt_attr
                        if max_val >= current_val:
                            max_val=max_val
                            max_attr_nme=max_attr_nme
                        else:
                            max_val=current_val
                            max_attr_nme=min_max_cnt_attr

                        min_cnt_attr_lst.remove(min_max_cnt_attr)
                cnt_index = cnt_index + 1

            # address only one occurance
            if cnt_attr_occurance == 1:
                min_val = 0

            total_error = total_error + min_val
            #Build descions of elements based on max occurance
            attr_element_prediction.update({cnt_ittr:max_attr_nme})
            #print('Min value %s %d , Max vlue %s %d ' % (min_attr_nme, min_val,max_attr_nme, max_val))
    #print total_error

    dict_attr_details ={attrname:{'header':{'attribute':attrname,'numofelements':len(origin_attr_val),'error':total_error},
                          'attrresult':attr_element_prediction
                         }
                        }
    return dict_attr_details

#Method will pick the best attribute and it details in jason format
def getbestattribute(p_detail_result_attributes):
    # finding best attribute
    print('Picking the best attribute ......')
    best_key=''
    current_min_error=-1
    current_cnt_elements=-1
    for v_key in p_detail_result_attributes[0].keys():
        #print(v_key,detail_result_attributes[0][v_key]['header']['error'])
        if current_min_error == -1:
            best_key=v_key
            current_min_error=p_detail_result_attributes[0][v_key]['header']['error']
            current_cnt_elements = p_detail_result_attributes[0][v_key]['header']['numofelements']
        elif p_detail_result_attributes[0][v_key]['header']['error'] < current_min_error:
            best_key = v_key
            current_min_error = p_detail_result_attributes[0][v_key]['header']['error']
            current_cnt_elements = p_detail_result_attributes[0][v_key]['header']['numofelements']
        elif p_detail_result_attributes[0][v_key]['header']['error'] > current_min_error:
            best_key = best_key
            current_min_error = current_min_error
            current_cnt_elements=current_cnt_elements
        elif p_detail_result_attributes[0][v_key]['header']['error'] == current_min_error:
            if p_detail_result_attributes[0][v_key]['header']['numofelements'] > current_cnt_elements:
                best_key = v_key
                current_min_error = p_detail_result_attributes[0][v_key]['header']['error']
                current_cnt_elements = p_detail_result_attributes[0][v_key]['header']['numofelements']

    #formating attribute values
    detail_result_attributes_format=copy.deepcopy(p_detail_result_attributes)
    attrresult_format=dict((ky,val[len(ky):]) for ky,val in p_detail_result_attributes[0][best_key]['attrresult'].items())
    detail_result_attributes_format[0][best_key]['attrresult']=attrresult_format


    #only for prinitng
    attrresult_format_print=dict((ky,val[len(ky):].lower().replace('yes','Play').replace('no','No Play')) for ky,val in p_detail_result_attributes[0][best_key]['attrresult'].items())
    detail_result_attributes_print=copy.deepcopy(p_detail_result_attributes)
    detail_result_attributes_print[0][best_key]['attrresult']=attrresult_format_print

    print('Most accurate Attribute is %s ' %(best_key))
    #print(detail_result_attributes_print)
    #return [detail_result_attributes_print[0][best_key]]
    return [detail_result_attributes_format[0][best_key]]

#main method to build the model
def build_model(trainFile):
    if not os.path.isfile(trainFile):
        print('File %s not exist !! . Exiting program'%(trainFile))
        return 1
    data=pd.read_csv(trainFile)

    OutCome_header=data.columns.values[len(data.columns.values)-1]
    #print (OutCome_header)
    detail_result_attributes={}
    print('Calculating accurancy of attributes ......')
    for attribute in data.columns.values:
        # skip outcome data process
        if attribute != OutCome_header:
            # if data is non numeric then call cat function
            if attribute[0:4].lower() == 'cat_':
                attribute_result=cat_attribute_accuracy(attribute,data[attribute],data[OutCome_header])
                #print ('Detail Result of attribute %s is %s' %(attribute,attribute_result))
                detail_result_attributes.update(attribute_result)
                #break;
        else:
            #print('Skipping OutCome attribute process')
            pass

    detail_result_attributes=[detail_result_attributes]
    #Pick the best attribute details
    return getbestattribute(detail_result_attributes)

#fill the test data using optimal rule
def live(p_livedatafile,p_optimal_rule):
    if not os.path.isfile(p_livedatafile):
        print('File %s not exist !! . Exiting program'%(p_livedatafile))
        return 1
    best_attribute=p_optimal_rule[0]['header']['attribute']
    livedata = open(p_livedatafile,'r')
    outlivedata = open('out_'+p_livedatafile, 'w')
    i = 0
    for ln in livedata:
        lnsplit = ln.split(',')
	#print(lnsplit)
        if i == 0 :
            attrIndex=[i for i, j in enumerate(lnsplit) if j == best_attribute]
            outcomeIndex=[x for x, y in enumerate(lnsplit) if y == 'OutCome\n']
        else:
            lnsplit[outcomeIndex[0]]=p_optimal_rule[0]['attrresult'][lnsplit[attrIndex[0]]]+'\n'
        i = i + 1
        outlivedata.write(','.join(lnsplit))
    livedata.close()
    outlivedata.close()

#optimal_rule=build_model('1rdata.csv')

#live('1rdata_test.csv',optimal_rule)
