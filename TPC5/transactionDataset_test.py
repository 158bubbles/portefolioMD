import unittest
from apriori import TransactionDataset

import unittest

class TransactionDatasetTestCase(unittest.TestCase):
    def setUp(self):
        # Create sample transaction dataset for testing
        self.transactions = [
            {1, 2, 3},
            {2, 3, 4},
            {1, 3, 4},
            {2, 3},
            {1, 2, 4},
        ]
        self.dataset = TransactionDataset(self.transactions)

    def test_frequent_items(self):
        expected_freq_items = [(2, 4), (3, 4), (1, 3), (4, 3)]
        self.assertEqual(self.dataset.freq_items, expected_freq_items)

    def test_unique_items(self):
        expected_unique_items = [1, 2, 3, 4]
        self.assertEqual(self.dataset.items, expected_unique_items)

if __name__ == '__main__':
    unittest.main()
