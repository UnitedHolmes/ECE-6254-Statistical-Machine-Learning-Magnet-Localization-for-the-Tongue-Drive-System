# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:26:21 2017

@author: NFLS_UnitedHolmes
"""

import xlsxwriter

workbook = xlsxwriter.Workbook('hello.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write('A1', 'Hello world')

workbook.close()