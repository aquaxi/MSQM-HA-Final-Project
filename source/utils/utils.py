#!/usr/bin/env python
# coding: utf-8

# In[ ]:

def direct_match(keyword, document):
  """ Count of substring keyword matches in a document

  This function counts the number of keyword matches in a document.
  This is a substring match and is case-sensitive.
  Example if keyword = 'tax' and document = 'tax on taxes' then
  the number of matches would be = 2.

  Args:
    keyword  (str): String to match on
    document (str): Document searched for matches

  Returns:
    int: Count of keyword matches in document
  """
  return document.count(keyword)