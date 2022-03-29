import sys
import os.path
import pandas as pd


def read_all_data(data_path, bill_col_idx):
    raw_data = pd.read_excel(data_path, header=None)
    raw_data.fillna(0)
    res = []

    for r in list(raw_data.iloc[:, bill_col_idx]):
        if isinstance(r, str):
            res.append(float(''.join(r.split(','))))
        elif isinstance(r, float):
            res.append(r)
    raw_data[bill_col_idx] = res
    return raw_data


def read_data(data_path, bill_col_idx):
    raw_data = pd.read_excel(data_path, header=None)
    raw_data.fillna(0)
    res = []

    for r in list(raw_data.iloc[:, bill_col_idx]):
        if isinstance(r, str):
            res.append(float(''.join(r.split(','))))
        elif isinstance(r, float):
            res.append(r)

    return res

    # return [float(''.join(r.split(','))) for r in list(raw_data.iloc[:, bill_col_idx])]

    # with open(data_path, 'r') as f:
    #     # for csv file->split(','), for exl file->split('\t')
    #     raw_data = [r.strip().split('\t') for r in f.readlines()]
    #     return [float(b[bill_col_idx]) for b in raw_data]


def check_bills(account_bill_file, acc_bill_col,
                record_bill_file, rec_bill_col, file_flag):
    acc_bill = sorted(read_data(account_bill_file, acc_bill_col),
                      reverse=True)
    rec_bill = sorted(read_data(record_bill_file, rec_bill_col),
                      reverse=True)
    if len(acc_bill) > len(rec_bill):
        print("银行流水数量多于记账数目")
    elif len(acc_bill) < len(rec_bill):
        print("记账数目多于银行流水数量")
    res = []
    bank_res_set = []
    record_res_set = []
    acc_i, rec_j = 0, 0
    while (acc_i < len(acc_bill)) and (rec_j < len(rec_bill)):
        if float(acc_bill[acc_i]) == float(rec_bill[rec_j]):
            acc_i += 1
            rec_j += 1
        elif float(acc_bill[acc_i]) > float(rec_bill[rec_j]):
            print("account bill suspected error:{}".format(acc_bill[acc_i]))
            res.append(["account bill suspected error", acc_bill[acc_i]])
            if acc_bill[acc_i] not in bank_res_set:
                bank_res_set.append(acc_bill[acc_i])
            acc_i += 1
        elif float(acc_bill[acc_i]) < float(rec_bill[rec_j]):
            print("record bill suspected error:{}".format(rec_bill[rec_j]))
            res.append(["record bill suspected error", rec_bill[rec_j]])
            if rec_bill[rec_j] not in record_res_set:
                record_res_set.append(rec_bill[rec_j])
            rec_j += 1
    # res = pd.DataFrame(res)
    # res.columns = ["error type", "error money"]
    # 如果两边比较中出现某一边的索引溢出而另一边还没有比完的情况，需要把没有比完的那一边的数据拿到结果中
    if (acc_i < len(acc_bill)) or (rec_j < len(rec_bill)):
        try:
            if rec_bill[rec_j]:
                res += rec_bill[rec_j:]
                for rec_over in rec_bill[rec_j:]:
                    if rec_over not in record_res_set:
                        record_res_set.append(rec_over)
            
        except:
            if acc_bill[acc_i]:
                res += acc_bill[acc_i:]
                for acc_over in acc_bill[acc_i:]:
                    if acc_over not in bank_res_set:
                        bank_res_set.append(acc_over)
    
    bank_df = read_all_data(account_bill_file, acc_bill_col)
    record_df = read_all_data(record_bill_file, rec_bill_col)
    res_b = bank_df[bank_df[acc_bill_col] == bank_res_set[0]]
    res_r = record_df[record_df[rec_bill_col] == record_res_set[0]]
    for i in bank_res_set[1:]:
        res_b = res_b.append(bank_df[bank_df[acc_bill_col] == i])
    # res_b = pd.DataFrame(res_b)
    res_b.to_excel(os.path.dirname(account_bill_file) + r"\{}_yinhangliushui_suspect_error.xls".format(file_flag))

    for i in record_res_set[1:]:
        res_r = res_r.append(record_df[record_df[rec_bill_col] == i])
    # res_r = pd.DataFrame(res_r)
    res_r.to_excel(os.path.dirname(account_bill_file) + r"\{}_sanlianzhang_suspect_error.xls".format(file_flag))
    # res.to_excel(os.path.dirname(account_bill_file)+r"\{}_suspect_error.xls".format(file_flag))


if __name__ == "__main__":
    bank_in_col = 3
    bank_out_col = 4
    three_in_col = 5
    three_out_col = 6
    print("请先把流水和三联账的表头给删除！！！")
    flag = input("请小阿宝输入 income 或者 payout :\n")
    account_file = input("请小阿宝输入 无表头银行流水文件路径\n")
    record_file = input("请小阿宝输入 无表头三联账文件路径\n")
    confirm = input("请确认 银行流水文件中 收入 和 支出 为 第3和第4列 \n以及请确认 三联账文件中 收入 和 支出 为 第5和第6列 \n 请输入：yes或者no\n")
    # account_file = sys.argv[1]
    # record_file = sys.argv[2]
    # flag = sys.argv[3]
    if confirm == 'no':
        bank_in_col = int(input("银行流水文件中 收入 为第几列：\n"))
        bank_out_col = int(input("银行流水文件中 支出 为第几列：\n"))
        three_in_col = int(input("三联账文件中 收入 为 第几列：\n"))
        three_out_col = int(input("三联账文件中 支出 为 第几列：\n"))
    if flag == "income":
        account_col = bank_in_col
        record_col = three_in_col
    else:
        account_col = bank_out_col
        record_col = three_out_col

    check_bills(account_file, account_col,
                record_file, record_col, flag)
    # check_bills(r"C:\Users\liyihang\Desktop\account1.xlsx", 4,
    #             r"C:\Users\liyihang\Desktop\record1.xlsx", 5)
