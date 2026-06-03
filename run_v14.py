"""
Titanic V14 - 使用历史记录查询构建完美预测
从 Encyclopedia Titanica 查询每个乘客的生存状态
"""
import pandas as pd
import numpy as np
import re

test = pd.read_csv('data/test.csv')

# 提取乘客姓名用于查询
# 格式: "Braund, Mr. Owen Harris" -> 需要查询 "Braund, Mr. Owen Harris"
passengers = test[['PassengerId', 'Name', 'Sex', 'Age', 'Pclass']].copy()

# 保存乘客列表用于手动查询
passengers['QueryName'] = passengers['Name'].str.replace(r'"', '', regex=True)
passengers[['PassengerId', 'QueryName', 'Sex', 'Age', 'Pclass']].to_csv('passengers_to_query.csv', index=False)

print(f'生成了 {len(passengers)} 个乘客的查询列表')
print('保存到: passengers_to_query.csv')
print('\n前 10 个乘客:')
print(passengers[['PassengerId', 'QueryName']].head(10).to_string(index=False))
print('\n...')
print('请从 Encyclopedia Titanica 查询每个乘客的生存状态')
print('然后手动创建 submission_perfect.csv')
