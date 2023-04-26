from collections import defaultdict, Counter
from typing import Tuple, List
from math import log2

import sys
sys.setrecursionlimit(int(1e9))

def get_most_frequent_label(label_list) -> str:
    """ 최다득표로 레이블 선정 """
    return Counter(label_list).most_common(1)[0][0]


def info(dataset) -> float:
    """ info(D) 계산식 """
    label_list = [row[-1] for row in dataset]
    label_counter = Counter(label_list)
    result = 0.0
    for val in label_counter.values():
        p = val / len(label_list)
        result += p * log2(p)
    return -result


def split_with_category(dataset, feature_idx) -> Tuple[List, List[List]]:
    """ feature의 카테고리 별로 배열 분리 """
    dic = defaultdict(list)
    for row in dataset:
        dic[row[feature_idx]].append(row)
    return list(dic.keys()), list(dic.values())


def get_best_feature_idx(dataset, visited) -> int:
    """ Best feature의 인덱스 리턴 """
    infoD = info(dataset)
    best_gain_ratio = -1
    best_feature_idx = -1

    for feature_idx in range(len(dataset[0]) - 1):
        if visited[feature_idx]:
            continue

        _, split_dataset_arr = split_with_category(dataset, feature_idx)
        
        infoA, split_info = 0, 0
        for split_dataset in split_dataset_arr:
            # val = |Di| / |D|
            val = len(split_dataset) / len(dataset)

            # (|Di| / |D|) * info(Di)
            infoA += val * info(split_dataset)

            # - (|Di| / |D|) * log2(|Di| / |D|)
            split_info -= val * log2(val)

        gain_ratio = (infoD - infoA) / split_info
        if best_gain_ratio < gain_ratio:
            best_gain_ratio = gain_ratio
            best_feature_idx = feature_idx

    return best_feature_idx


def build_tree(dataset, features, all_category, visited):
    label_list = [row[-1] for row in dataset]

    # 종료 컨디션 1: 레이블이 하나밖에 남지 않았을 때
    if len(label_list) == label_list.count(label_list[0]):
        return label_list[0]

    # 종료 컨디션 2: 더 이상 사용할 컬럼이 남지 않았을 때
    if all(visited):
        return get_most_frequent_label(label_list)

    # Gain Ratio 가 가장 큰 인덱스 설정
    best_feature_idx = get_best_feature_idx(dataset, visited)
    best_feature_name = features[best_feature_idx]
    visited[best_feature_idx] = True

    tree = { best_feature_name: {} }

    # 해당 인덱스로 분리하고 분리된 걸 recursive
    split_keys, split_dataset_arr = split_with_category(dataset, best_feature_idx)
        
    for key, split_dataset in zip(split_keys, split_dataset_arr):
        tree[best_feature_name][key] = build_tree(split_dataset, features, all_category, visited[:])
    
    if len(split_keys) != len(all_category[best_feature_idx]):
        for name in all_category[best_feature_idx]:
            if name not in tree[best_feature_name]:
                tree[best_feature_name][name] = get_most_frequent_label(label_list)

    return tree


def throw_error_with_message(message : str) -> None:
    ''' 에러 처리를 위한 함수 '''
    print(f'Error: {message}')
    sys.exit(1)


def run():
    ''' 초기 세팅을 위한 함수 '''
    def get_variables_from_args():
        ''' 프로그램 인자로부터 변수 세팅 '''

        argv = sys.argv
        if len(argv) != 4:
            throw_error_with_message("usage: python dt.py (train file name) (test file name) (output file name)")
        return argv[1], argv[2], argv[3]

    def get_data(path, is_test):
        with open(path) as f:
            arr = [line.strip().split("\t") for line in f.readlines()]
            columns = arr[0][0:-1] if not is_test else arr[0]
            dataset = arr[1:]
            return dataset, columns, arr[0][-1]

    train_file_name, test_file_name, output_file_name = get_variables_from_args()
    dataset, train_columns, label = get_data(train_file_name, False)
    testset, test_columns, _ = get_data(test_file_name, True)

    all_category = [set() for _ in range(len(train_columns))]
    for row in dataset:
        for j in range(len(train_columns)):
            all_category[j].add(row[j])

    label_data = []
    for row in dataset:
        label_data.append(row[-1])
    miss_label = Counter(label_data).most_common(1)[0][0]

    tree = build_tree(dataset, train_columns, all_category, [False for _ in range(len(train_columns))])

    answer_arr = []
    for row in testset:
        point = tree
        while type(point) == dict:
            for j, col in enumerate(test_columns):
                if col == list(point.keys())[0]:
                    point = point[col]
                    if row[j] in point:
                        point = point[row[j]]
                        if type(point) != dict:
                            answer_arr.append(point)
                            break
                    else:
                        answer_arr.append(miss_label)
                        point = 'fin'
                        break

    with open(output_file_name, 'w') as f:
        f.write('\t'.join(test_columns + [label]) + '\n')
        for row, ans in zip(testset, answer_arr):
            line = '\t'.join(row + [ans]) + '\n' 
            f.write(line)   


if __name__ == '__main__':
    run()
