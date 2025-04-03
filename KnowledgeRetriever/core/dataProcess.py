import json
def data4json(data_o: dict, file_name: str, title = "ZZZZ"):
    data = {}
    Counter = 0
    Subject = f"{file_name}_{title}"

    for title, content in data_o.items():
        Id = f"{Subject}_{str(Counter).rjust(3, '0')}"
        item = {"title": title,
                "clusters": []}

        # 文本分割
        comments = content.split("\n")
        for comment in comments:
            if not comment: continue
            item["clusters"].append({"comments": comment})

        data[Id] = item

        Counter += 1

    with open(f"data/{file_name}.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def regulaProcess(data_path, Topictitle):
    with open(data_path, "r", encoding="utf-8") as file:
        data = file.readlines()

    Btitle = ""
    Ztitle = ""
    Ttitle = ""
    content = ""
    title = ""
    GCDdata = {}

    for item in data:
        item = item.strip()
        if not item:
            continue
        if "编" in item[:3]:
            Btitle = item[:3]
            continue
        if "章" in item[:4]:
            Ztitle = item[:4].strip()
            continue
        if item[0] == "第" and "条" in item:
            if title and content:
                GCDdata[title] = content

            temp = item.split("条")
            Ttitle = temp[0] + "条"
            content = "条".join(temp[1:]).strip()+"\n"
        else:
            content += item

        title = f"{Topictitle}{Btitle}{Ztitle}{Ttitle}"

    GCDdata[title] = content
    return GCDdata

def standardProcess(data_path, Topictitle):
    with open(data_path, "r", encoding="utf-8") as file:
        data = file.readlines()

    T = ""
    Ttitle = ""
    result_data = {}

    for item in data:
        Xtitle = ""
        X = ""
        item = item.strip()
        if "第" in item[:4] and "条" in item[:4]:
            T = item
            temp = item.split("条")
            Ttitle = temp[0] + "条"
        if "（" in item[:4] and "）" in item[:4]:
            X = item
            temp = item.split("）")
            Xtitle = "第" + temp[0][1:] + "项"

        content = T+X

        title = f"{Topictitle}{Ttitle}{Xtitle}"
        result_data[title] = content

    return result_data


import pandas as pd
def caseProcess(data_path, Topictitle):
    data = pd.read_excel(data_path).to_dict('records')
    caseData = {}
    caseData[Topictitle] = '\n'.join([str(case) for case in data])
    return caseData

if __name__ == '__main__':
    # data_o = regulaProcess("data/国有企业领导人员廉洁从业若干规定","《国有企业领导人员廉洁从业若干规定》")
    # data_o = caseProcess("data/送礼案例.xlsx", "违规送礼")
    data_o = standardProcess("data/《煤矿重大事故隐患判定标准》","《煤矿重大事故隐患判定标准》")
    data4json(data_o, "《煤矿重大事故隐患判定标准》", title="GZPD")

