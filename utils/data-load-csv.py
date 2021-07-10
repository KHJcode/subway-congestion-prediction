import csv

dataset_path = './dataset/subway.csv'

f = open(dataset_path, 'r', encoding='cp949')
sub_data = csv.reader(f, delimiter=',')
next(sub_data)

subway = {}
kinds = ['1호', '2호', '3호', '4호', '5호', '6호', '7호', '8호', '9호', '분당', '경의', '공항철도 1호', '경간', '경부', '경원', '경인', '경춘', '과천', '분당', '수인', '안산', '우이신설', '일산', '장항', '중앙']

print('데이터 로딩중...')

for row in sub_data:
    for item in kinds:
        key = item + '선'
        if key in row[1]:
            if key not in subway.keys():  subway[key] = []
            subway[key].append(row)
            break

print('로딩 완료')

def getSubwayData(key):    return subway[key]
