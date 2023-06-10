import sys
import re
sys.setrecursionlimit(100000)
from collections import defaultdict
from typing import Set, List

TOTAL_DICT = dict()
LABEL = defaultdict(str)
# VISIT = 
NEIGHBORS_OF_CORE = defaultdict(list)
INPUT_FILE_NAME = None; K = None; EPS = None; MIN_POINTS = None
RETURN_ARRAY = []
INPUT_FILE_NUM = None

def throw_error_with_message(message : str) -> None:
    ''' 에러 처리를 위한 함수 '''
    print(f'Error: {message}')
    sys.exit(1)


def set_all():
    ''' 초기 세팅을 위한 함수 '''
    def set_variables_from_args():
        ''' 프로그램 인자로부터 변수 세팅 '''
        global INPUT_FILE_NAME, EPS, MIN_POINTS, INPUT_FILE_NUM, K

        argv = sys.argv
        if len(argv) != 5:
            throw_error_with_message("usage: python clustering.py (input file name) (n) (Eps) (Minpts)")
        try:
            INPUT_FILE_NAME = argv[1]; K = int(argv[2]); EPS = float(argv[3]); MIN_POINTS = float(argv[4])
            matched = re.compile('\d+').findall(INPUT_FILE_NAME.split('.')[0])
            if len(matched) == 0:
                INPUT_FILE_NUM = 'N'
            else:
                INPUT_FILE_NUM = matched[0]
        except ValueError:
            throw_error_with_message("Invalid Arguments")

    def set_points_from_file():
        ''' 입력 파일로부터 변수 세팅 '''
        global N
    
        TOTAL_DICT.clear()

        with open(INPUT_FILE_NAME, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                object_id, x, y = map(float, line.rstrip().split())
                TOTAL_DICT[int(object_id)] = (x, y)
            N = len(TOTAL_DICT)

    set_variables_from_args()
    set_points_from_file()


def is_neighbors(i, j) -> bool:
    dist = ((TOTAL_DICT[i][0] - TOTAL_DICT[j][0]) ** 2 + (TOTAL_DICT[i][1] - TOTAL_DICT[j][1]) ** 2) ** 0.5
    return dist < EPS


def range_query(i):
    neighbors = []
    for j in TOTAL_DICT.keys():
        if is_neighbors(i, j):
            neighbors.append(j)
    return neighbors


def expand_cluster(i: int, cluster_id: int, neighbors: list):
    LABEL[i] = cluster_id

    k = 0
    while True:
        try:
            j = neighbors[k]
        except:
            pass

        if LABEL[j] == '':
            n = range_query(j)
            if len(n) > MIN_POINTS:
                neighbors.extend(n)

        LABEL[j] = str(cluster_id)

        k += 1
        if len(neighbors) < k:
            return


def dbscan():
    cluster_id = 0
    for p in TOTAL_DICT.keys():
        if LABEL[p] != '':
            continue

        neighbors = range_query(p)
        if len(neighbors) < MIN_POINTS:
            LABEL[p] = 'Noise'
        else:
            expand_cluster(p, cluster_id, neighbors)
            cluster_id += 1

    class_to_list = defaultdict(list)
    for idx, cluster_id in LABEL.items():
        class_to_list[cluster_id].append(idx)
    
    for arr in class_to_list.values():
        RETURN_ARRAY.append(sorted(arr))
    
    RETURN_ARRAY.sort(key=lambda x: len(x), reverse=True)


def write_output():
    for i in range(K):
        lines = []
        with open(f"input{INPUT_FILE_NUM}_cluster_{i}.txt", "w") as f:
            for data in RETURN_ARRAY[i]:
                lines.append(str(data) + '\n')
            f.writelines(lines)

if __name__ == '__main__':
    set_all()
    dbscan()
    write_output()
    