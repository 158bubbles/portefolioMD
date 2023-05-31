import unittest
from apriori import Apriori
from apriori import TransactionDataset
from unittest.mock import MagicMock

class AprioriTestCase(unittest.TestCase):
    def setUp(self):
        # Create a mock TransactionDataset object for testing
        
        transactions = [    ['lapis', 'marcador'],
                    ['lapis', 'caneta', 'marcador'],
                    ['caneta', 'marcador'],
                    ['caneta', 'caderno', 'marcador', 'lapis'],
                    ['lapis', 'caderno'],
                    ['caneta', 'marcador'],
                    ['caderno', 'marcador']
               ]

        transaction_dataset = TransactionDataset(transactions)
        self.apriori = Apriori(transaction_dataset, 0.3)
        
    def test_apriori_algorithm(self):
        expected_freq_itemsets = {

            frozenset({'marcador'}): 6, 
            frozenset({'lapis'}): 4, 
            frozenset({'caneta'}): 4, 
            frozenset({'caderno'}): 3, 
            frozenset({'lapis', 'marcador'}): 3, 
            frozenset({'caneta', 'marcador'}): 4
        }

        self.assertEqual(self.apriori.freq_Items, expected_freq_itemsets)

    def test_association_rules(self):
        expected_rules = [
            ((frozenset({'lapis'}), frozenset({'marcador'})), 0.75), 
            ((frozenset({'marcador'}), frozenset({'lapis'})), 0.5), 
            ((frozenset({'caneta'}), frozenset({'marcador'})), 1.0), 
            ((frozenset({'marcador'}), frozenset({'caneta'})), 0.6666666666666666)
        ]

        self.apriori.association_rules(0.5)
        for rule in expected_rules:
            rule_found = False
            for antecedent, consequent in self.apriori.rules.items():
                
                if (antecedent, consequent) == rule or (antecedent, consequent) == (rule[1], rule[0]):
                    rule_found = True
                    break
            self.assertTrue(rule_found, f"Rule {rule} not found in self.apriori.rules")


        

    # Add more test methods to cover other functionality of the Apriori class

if __name__ == '__main__':
    unittest.main()

