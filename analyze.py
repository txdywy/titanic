"""
Titanic V11 - 规则 + 模型混合策略
基于领域知识: "Women and children first" + 阶层差异
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 分析训练集模式
print('=== Training Data Patterns ===')
print('Survival by Sex:')
print(train.groupby('Sex')['Survived'].mean())
print('\nSurvival by Pclass:')
print(train.groupby('Pclass')['Survived'].mean())
print('\nSurvival by Sex + Pclass:')
print(train.groupby(['Sex','Pclass'])['Survived'].mean())
print('\nSurvival by Sex + Pclass + Age group:')
train['AgeGroup'] = pd.cut(train['Age'], bins=[0,12,18,60,100], labels=['child','teen','adult','elderly'])
print(train.groupby(['Sex','Pclass','AgeGroup'], observed=True)['Survived'].agg(['mean','count']))

# 分析 3 等舱女性
female_3 = train[(train['Sex']=='female') & (train['Pclass']==3)]
print(f'\n3rd class female survival: {female_3["Survived"].mean():.3f} ({len(female_3)} total)')
print(f'3rd class female by Embarked:')
print(female_3.groupby('Embarked')['Survived'].agg(['mean','count']))

# 分析 1 等舱男性
male_1 = train[(train['Sex']=='male') & (train['Pclass']==1)]
print(f'\n1st class male survival: {male_1["Survived"].mean():.3f} ({len(male_1)} total)')
