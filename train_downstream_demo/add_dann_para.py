input_file = "/data/zhaohaishu/Codes/emotion2vec_upload/train_downstream_demo/train.emo"
output_file = "/data/zhaohaishu/Codes/emotion2vec_upload/train_downstream_demo/train.dann"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.rstrip("\n")
        fout.write(f"{line} Human\n")
