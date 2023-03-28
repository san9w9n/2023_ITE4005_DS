# Ck: Candidate itemset of size k
# Lk: frequent itemset of size k
# L1= {frequent items};

# for(k = 1; Lk != None; k++) do begin
#   Ck+1 = candidates generated from Lk
#   for each transaction t in database do
#       increment the count of all candidates in Ck+1 that are contained in t
#       Lk+1 = candidates in Ck+1 with min_support
#   end
# return ∪kLk

from itertools import chain, combinations
import sys

from collections import defaultdict
from typing import Set, List, Dict, Tuple

min_sup: float = None
input_file_name: str = None
output_file_name: str = None
trxs: List[frozenset] = None
results: List[Tuple[frozenset, frozenset, float, float]] = None
item_set : Dict[int, Set[frozenset]] = {}
freq_dict: defaultdict[frozenset, int] = defaultdict(int)


def throw_error_with_message(message : str) -> None:
    print(f'Error: {message}')
    sys.exit(1)


def set_all():
    def set_variables_from_args():
        global min_sup, input_file_name, output_file_name

        argv = sys.argv
        if len(argv) != 4:
            throw_error_with_message("usage: python apriori.py (minimum support) (input file name) (output file name)")
        try:
            min_sup, input_file_name, output_file_name = float(argv[1]), argv[2], argv[3]
        except ValueError:
            throw_error_with_message("minimum support should be number(float, int ...)")

    def set_transactions_list_from_file():
        global trxs

        trxs = []
        with open(input_file_name, 'r', encoding='utf-8') as f:
            while True:
                raw_trx = f.readline()
                if not raw_trx:
                    break
                trxs.append(frozenset(list(map(str.strip, raw_trx.replace('\n', '').split('\t')))))

    set_variables_from_args()
    set_transactions_list_from_file()


def set_item_set_of_k(k: int):
    global item_set

    if k == 1:
        item_set[1] = { frozenset([c]) for trx in trxs for c in trx } 
    else:
        ret = []
        for i in item_set[k-1]:
            for j in item_set[k-1]:
                new_set = i.union(j)
                if len(new_set) == k:
                    ret.append(new_set)
        item_set[k] = set(ret)


def get_array_of_subset_pair(s: frozenset) -> List[Tuple[frozenset, frozenset]]:
    subsets = chain(*[combinations(s, i + 1) for i in range(len(s))])
    subsets_iter = map(frozenset, [subset for subset in subsets])

    ret = []
    for subset1 in subsets_iter:
        subset2 = s.difference(subset1)
        if len(subset2) > 0:
            ret.append((subset1, subset2))
    return ret


def remove_infrequent_item_from_item_set(k: int):
    global freq_dict, item_set

    local_dict: defaultdict[frozenset, int] = defaultdict(int)

    for item in item_set[k]:
        if len(item) != k:
            continue
        for trx in trxs:
            if item.issubset(trx):
                local_dict[item] += 1
                freq_dict[item] += 1

    for c, cnt in local_dict.items():
        sup = (cnt / len(trxs)) * 100
        if sup < min_sup:
            item_set[k].remove(c)


def start_apriori():
    global results

    k =  1
    while True:
        set_item_set_of_k(k)
        remove_infrequent_item_from_item_set(k)
        if len(item_set[k]) == 0:
            del item_set[k]
            break
        k += 1

    results = []
    for k, item in item_set.items():
        if k > 1:
            for s in item:
                support = freq_dict[s] / len(trxs)
                array_of_subset_pair = get_array_of_subset_pair(s)
                for subset1, subset2 in array_of_subset_pair:
                    sub_support = freq_dict[subset1] / len(trxs)
                    confidence = support / sub_support
                    results.append((subset1, subset2, round(support * 100, 2), round(confidence * 100, 2)))


def write_results():
    with open(output_file_name, "w", encoding='utf-8') as f:
        for result in results:
            str_result = [
                str(set(result[0])).replace("'", "").replace(" ", "").strip(),
                str(set(result[1])).replace("'", "").replace(" ", "").strip(),
                str(result[2]),
                str(result[3])
            ]
            f.write("\t".join(str_result) + "\n")


if __name__ == '__main__':
    set_all()

    start_apriori()

    write_results()
