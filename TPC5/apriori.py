"""
A classe TransactionDataset é usada para armazenar um conjunto de transações, em que cada transação é representada por uma lista de itens.

'frequent_items' conta a ocorrência de cada item nas transações e retorna uma lista de tuplas contendo o item e sua contagem.
"""

from typing import List, Set, Dict, Tuple

class TransactionDataset:
    def __init__(self, transactions: List[Set[int]]):
        self.transactions = transactions   #list of sets
        self.freq_items = self.frequent_items()   #frequent items in the transaction dataset
        self.items = self.unique_items()  #list of unique items

    #Method that calculates the frequency of items in 'transactions' and returns a list in descending order of frequency
    def frequent_items(self):
        counter = {}

        for t in self.transactions:
            for i in t:
                #if it exists increment, otherwise initialize counter
                if i in counter:
                    counter[i] += 1
                else:
                    counter[i] = 1

        #return a list of tuples
        freq_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        print("Frequent items: ", freq_items)
        return freq_items


    #Method that returns a list of unique items present in 'transactions'
    def unique_items(self):
        items = set()

        for t in self.transactions:
            for item in t:
                items.add(item)
        return list(items)

transactions = [    ['lapis', 'marcador'],
                    ['lapis', 'caneta', 'marcador'],
                    ['caneta', 'marcador'],
                    ['caneta', 'caderno', 'marcador', 'lapis'],
                    ['lapis', 'caderno'],
                    ['caneta', 'marcador'],
                    ['caderno', 'marcador']
               ]

transaction_dataset = TransactionDataset(transactions)
print("And these are the unique items:", transaction_dataset.items)

"""A classe Apriori tem como objetivo encontrar conjuntos frequentes de itens num conjunto de transações.

As regras de associação permitem identificar relações ou associações significativas entre diferentes itens. Assim, podemos afirmar que se o antecedente se verifica, entao ocorre o consequente com um certo valor de confiança.
"""

from collections.abc import Reversible
from typing import List, Set, Dict
from itertools import combinations
from collections import defaultdict

class Apriori:
    def __init__(self, transaction_dataset: TransactionDataset, min_support: float):
        self.transaction_dataset = transaction_dataset    #object TransactionDataset
        self.min_support = min_support   # minimum frequency of an itemset, threshold
        self.freq_Items = self.apriori_Algorithm()    #frequent itemsets that satisfy the minimum support
        self.rules = {}

    def apriori_Algorithm(self):
        freq_items = {}
        freq_itemsets = {}
        candidates = self.transaction_dataset.freq_items

        for candidate, counter in candidates:
            if counter / len(self.transaction_dataset.transactions) >= self.min_support:
                freq_items[frozenset([candidate])] = counter

            freq_itemsets.update(freq_items) 
        
        i = 2
        
        while True:
            candidates = self.get_candidate_itemsets(freq_itemsets, i)
            if len(candidates) == 0:
                break

            frequentes = self.get_frequent_itemsets(candidates)
            if len(frequentes) == 0:
                break

            freq_itemsets.update(frequentes)
            i += 1

        return freq_itemsets
    
    
    #Method that get candidate itemsets of length i 
    def get_candidate_itemsets(self, freq_itemsets, i):
        candidates = set()

        for itemset1 in freq_itemsets:
            for itemset2 in freq_itemsets:

                if itemset1 != itemset2:
                    union = itemset1.union(itemset2)
                    if len(union) == i:
                        candidates.add(union)

        return candidates


    #Method that find frequent itemsets in a transaction dataset
    def get_frequent_itemsets(self, itemsets):
        item_counter = {}

        for transaction in self.transaction_dataset.transactions:
            for itemset in itemsets:

                if itemset.issubset(transaction):

                    #if the itemset is already in the dictionary, increments its count by 1
                    if itemset in item_counter:
                        item_counter[itemset] += 1
                    #if the itemset is not already in the dictionary, adds it with a count of 1 
                    else:
                        item_counter[itemset] = 1

        #filters out the itemsets whose support is less or equal than 'min_support'
        frequents = {itemset: c for itemset, c in item_counter.items() if
                         c / len(self.transaction_dataset.transactions) >= self.min_support}

        return frequents


    #Method that generates association rules based on frequent itemsets found
    def association_rules(self, min_confidence):
        rules = {}
        for itemset in self.freq_Items:
            for item in itemset:

                antecedent = frozenset([item])
                consequent = itemset - antecedent

                if(len(consequent)>0):
                    confidence = self.freq_Items[itemset] / self.freq_Items[antecedent]
                    if confidence >= min_confidence:
                        rules[(antecedent, consequent)] = confidence

        #add the new rules to 'rules'
        self.rules = rules

transactions = [    ['lapis', 'marcador'],
                    ['lapis', 'caneta', 'marcador'],
                    ['caneta', 'marcador'],
                    ['caneta', 'caderno', 'marcador', 'lapis'],
                    ['lapis', 'caderno'],
                    ['caneta', 'marcador'],
                    ['caderno', 'marcador']
               ]

transaction_dataset = TransactionDataset(transactions)
apriori = Apriori(transaction_dataset, 0.3)
print("Frequent itemsets:", apriori.freq_Items)

print("\n")

apriori.association_rules(0.5)
print("Association rules:")
for (antecedent, consequent), confidence in apriori.rules.items():
    print(f"{antecedent} => {consequent}: {confidence}")