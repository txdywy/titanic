"""
Titanic V13 - 基于历史记录的精确预测
"""
import pandas as pd
import numpy as np

test = pd.read_csv('data/test.csv')

# 基于历史记录的规则
# 关键洞察: 训练集中的模式非常清晰
# 1. 1/2等舱女性: 95%+ 存活
# 2. 3等舱女性: 50% 存活
# 3. 儿童: 高存活率
# 4. 1等舱男性: 37% 存活

# 创建多个规则变体
predictions = {}

# 变体 1: 保守规则 (基于训练集统计)
pred1 = np.zeros(len(test), dtype=int)
pred1[test['Sex'] == 'female'] = 1  # 所有女性
pred1[(test['Sex'] == 'male') & (test['Age'] < 12)] = 1  # 男性儿童
pred1[(test['Sex'] == 'male') & (test['Pclass'] == 1) & (test['Age'] < 50)] = 1  # 1等舱年轻男性
predictions['conservative'] = pred1

# 变体 2: 激进规则 (更多人存活)
pred2 = np.zeros(len(test), dtype=int)
pred2[test['Sex'] == 'female'] = 1
pred2[(test['Sex'] == 'male') & (test['Age'] < 15)] = 1
pred2[(test['Sex'] == 'male') & (test['Pclass'] == 1)] = 1
pred2[(test['Sex'] == 'male') & (test['Pclass'] == 2) & (test['Age'] < 10)] = 1
predictions['aggressive'] = pred2

# 变体 3: 性别基线 + 儿童
pred3 = np.where(test['Sex'] == 'female', 1, 0)
pred3[test['Age'] < 12] = 1
predictions['gender+child'] = pred3

# 变体 4: 性别基线 + 1等舱
pred4 = np.where(test['Sex'] == 'female', 1, 0)
pred4[test['Pclass'] == 1] = 1
predictions['gender+1st'] = pred4

# 变体 5: 混合规则
pred5 = np.where(test['Sex'] == 'female', 1, 0)
pred5[(test['Sex'] == 'male') & (test['Age'] < 12)] = 1
pred5[(test['Sex'] == 'male') & (test['Pclass'] == 1) & (test['Age'] < 50)] = 1
pred5[(test['Sex'] == 'female') & (test['Pclass'] == 3) & (test['Age'] > 30)] = 0  # 3等舱大龄女性
predictions['mixed'] = pred5

print('=== 预测统计 ===')
for name, pred in predictions.items():
    print(f'{name:15s}: {pred.sum():3d} survive / {418-pred.sum():3d} die')

# 保存所有变体
for name, pred in predictions.items():
    sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred})
    sub.to_csv(f'submission_{name}.csv', index=False)
    print(f'Saved submission_{name}.csv')
