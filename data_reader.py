import numpy as np
import csv

def process_data(data_str,dict=None):
    if(dict is None):
        return np.array([float(data_str)])
    if(len(dict)==1):
        return None
    if(len(dict)==2):
        return np.array([float(dict.index(data_str))])
    if(len(dict)>2):
        label=np.zeros(len(dict))
        label[dict.index(data_str)]=1.0
        return label


def process_number_row(row,division_part=10):
    data=np.zeros((len(row),division_part))
    thresh=np.zeros(division_part+1)
    min_number=data.min()
    distance=data.max()-data.min()
    distance=distance/division_part
    for i in range(0,11):
        thresh[i]=min_number+distance*i
    for j in range(0,len(row)):
        c_data=row[j]
        for i in range(0,10):
            if(c_data>=thresh[i] and c_data<thresh[i+1]):
                c_data[j,i]=1.0
                break
    return data


def process_number_to_label(matrix,division_part=10):
    logit_matrix=None
    for i in range(0,matrix.shape[1]):
        data=process_number_row(matrix[:,i])
        if(logit_matrix is None):
            logit_matrix=data
        else:
            logit_matrix=np.concatenate((logit_matrix,data),axis=1)

    return logit_matrix


def get_dict(file_path,key,csv_split):
    file=open(file_path,'rb')
    reader=csv.DictReader(file,delimiter=csv_split)
    dict=[]
    for row in reader:
        item=row[key]
        if(item not in dict):
            dict.append(item)
    file.close()
    return dict


def process_colume(file_path,dict,key,csv_split):
    file=open(file_path,'rb')
    reader=csv.DictReader(file,delimiter=csv_split)
    colume=[]
    for row in reader:
        colume.append(process_data(row[key],dict))
    colume=np.array(colume)
    file.close()
    return colume


def process_all(reader,value_key_list,string_key_list,csv_split=";"):
    matrix=None
    for key in value_key_list:
        colume=process_colume(reader,None,key,csv_split)
        if(matrix is None):
            matrix=colume
        else:
            matrix=np.concatenate((matrix,colume),axis=1)

    for key in string_key_list:
        dict=get_dict(reader,key,csv_split)
        colume=process_colume(reader,dict,key,csv_split)
        if(matrix is None):
            matrix=colume
        else:
            matrix=np.concatenate((matrix,colume),axis=1)
    return matrix


def read_bank_data(file_path="bank-additional.csv"):
    value_key_list=['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','euribor3m','nr.employed']
    string_key_list=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']
    matrix=process_all(file_path,value_key_list,string_key_list)
    return matrix[:,:-1],matrix[:,-1:]


def read_frog_data(file_path="Frogs_MFCCs.csv"):
    value_key_list=[]
    for i in range(1,10):
        key="MFCCs_ "+str(i)
        value_key_list.append(key)
    for i in range(10,23):
        key="MFCCs_"+str(i)
        value_key_list.append(key)
    data=process_all(file_path,value_key_list,string_key_list=[],csv_split=",")
    key_label=["Family"]
    label_family=process_all(file_path,[],key_label,csv_split=",")
    key_label=["Genus"]
    label_genus=process_all(file_path,[],key_label,csv_split=",")
    key_label=["Species"]
    label_species=process_all(file_path,[],key_label,csv_split=",")
    key_label=["RecordID"]
    label_record=process_all(file_path,[],key_label,csv_split=",")
    return data,label_family,label_genus,label_species,label_record


if __name__ == '__main__':
    data,label=read_bank_data()
    print(data)
    print(label)
    data,label_family,label_genus,label_species,label_record=read_frog_data()
    print(data)
    print(label_family)
    print(label_genus)
    print(label_species)
    print(label_record)
