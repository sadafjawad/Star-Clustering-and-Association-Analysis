#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 19:17:13 2024

@author:
"""
import pandas as pd
from apyori import apriori

#***** Association Analysis Problem *****
data = pd.read_csv(r'/Users/sadafjawad/Desktop/CPS844/Assignment2/6_class_csv.csv')

#Preprocessing
#keep categorical values like Star color, Spectral class, and Star type
data = data.drop(data.columns[:4], axis=1)
#convert star type from int to string
data['Star type'] = data['Star type'].map({
    0: 'Brown Dwarf', 
    1: 'Red Dwarf',
    2: 'White Dwarf',
    3: 'Main Sequence',
    4: 'Supergiant',
    5: 'Hypergiant'
})
#make all colors lowercase to keep consistency
data['Star color'] = data['Star color'].str.lower()
#make all colors use - in between to keep consistency
replacements = {
    ' ': '-',
}
data['Star color'] = data['Star color'].replace(replacements, regex=True)

#store as list of list in order to use apriori
dataList = [list(row) for row in data.values]
    
#association analysis using the apriori algorithm
itemsets = apriori(dataList, min_support = 0.1, min_confidence = 0.7)

ruleList = []
#add the rules in the rules list to print as a table
for itemset in itemsets:
    for rule in itemset.ordered_statistics:
        rule_str = '{{{}}} -> {{{}}}'.format(', '.join(rule.items_base), ', '.join(rule.items_add))
        support = itemset.support
        confidence = rule.confidence
        ruleList.append([rule_str, support, confidence])
        #print(f"{itemBase} -> {itemsAdd}, Support: {support}, Confidence: {confidence}")
rules_df = pd.DataFrame(ruleList, columns=['Rule', 'Support', 'Confidence'])
#rules_df.to_excel('/Users/sadafjawad/Desktop/CPS844/Assignment2/star_association_rules.xlsx', index=False)
