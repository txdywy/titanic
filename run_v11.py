"""
Titanic V11 - 多策略混合提交
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
train['is_train'] = 1
test['is_train'] = 0
test['Survived'] = np.nan
full = pd.concat([train, test], sort=False).reset_index(drop=True)

def engineer(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {'Mr':'Mr','Miss':'Miss','Mrs':'Mrs','Master':'Master',
                 'Dr':'Rare','Rev':'Rare','Col':'Rare','Major':'Rare',
                 'Mlle':'Miss','Ms':'Miss','Mme':'Mrs','Don':'Rare',
                 'Dona':'Rare','Lady':'Rare','Countess':'Rare',
                 'Jonkheer':'Rare','Sir':'Rare','Capt':'Rare'}
    df['Title'] = df['Title'].map(title_map).fillna('Rare')
    df['NameLen'] = df['Name'].apply(len)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FamilyCat'] = df['FamilySize'].map(lambda x: 'Single' if x==1 else 'Small' if x<=4 else 'Large')
    ticket_counts = df['Ticket'].value_counts()
    df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['CabinDeck'] = df['Cabin'].str[0].fillna('U')
    df['CabinNum'] = df['Cabin'].str.extract(r'(\d+)').astype(float)
    df['CabinCount'] = df['Cabin'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    for grp in [['Title','Pclass','Sex'], ['Title','Pclass']]:
        df['Age'] = df['Age'].fillna(df.groupby(grp)['Age'].transform('median'))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['IsChild'] = (df['Age'] < 12).astype(int)
    df['IsElderly'] = (df['Age'] > 60).astype(int)
    df['Age*Class'] = df['Age'] * df['Pclass']
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']
    df['FareLog'] = np.log1p(df['Fare'])
    df['FareBin'] = pd.qcut(df['Fare'], 5, labels=False, duplicates='drop')
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Sex*Pclass'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)
    df['Title*Pclass'] = df['Title'] + '_' + df['Pclass'].astype(str)
    for col in ['Embarked', 'Title', 'FamilyCat', 'CabinDeck', 'Sex*Pclass', 'Title*Pclass']:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        df = pd.concat([df, dummies], axis=1)
    return df

full_fe = engineer(full)

feature_cols = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'FamilySize', 'IsAlone', 'HasCabin', 'CabinNum', 'CabinCount',
    'IsChild', 'IsElderly', 'Age*Class', 'FarePerPerson', 'FareLog', 'FareBin',
    'TicketGroupSize', 'NameLen'
]
feature_cols += [c for c in full_fe.columns if c.startswith(('Embarked_','Title_','FamilyCat_','CabinDeck_','Sex*Pclass_','Title*Pclass_'))]

train_fe = full_fe[full_fe['is_train']==1]
test_fe = full_fe[full_fe['is_train']==0]
X = train_fe[feature_cols].fillna(0)
y = train_fe['Survived'].astype(int)
Xt = test_fe[feature_cols].fillna(0)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# V7 最佳模型
rf = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_split=6, min_samples_leaf=3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, min_samples_split=6, min_samples_leaf=3, random_state=42)
svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, C=5, kernel='rbf', random_state=42))])
lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, C=1, random_state=42))])
knn = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))])

v7 = VotingClassifier(
    estimators=[('rf',rf),('gb',gb),('svm',svm),('lr',lr),('knn',knn)],
    voting='soft'
)
v7.fit(X, y)
v7_probs = v7.predict_proba(Xt)[:, 1]

# 规则策略
test_data = test_fe.copy()

# 策略 A: V7 原版
pred_a = (v7_probs >= 0.5).astype(int)

# 策略 B: V7 + 儿童规则 (所有 <12 岁存活)
pred_b = pred_a.copy()
pred_b[test_data['Age'] < 12] = 1

# 策略 C: V7 + 1/2 等舱女性规则
pred_c = pred_a.copy()
pred_c[(test_data['Sex']==1) & (test_data['Pclass']<=2)] = 1

# 策略 D: V7 + 儿童 + 1/2等舱女性 + 3等舱S口岸女性死亡
pred_d = pred_a.copy()
pred_d[test_data['Age'] < 12] = 1  # 儿童存活
pred_d[(test_data['Sex']==1) & (test_data['Pclass']<=2)] = 1  # 1/2等舱女性
pred_d[(test_data['Sex']==1) & (test_data['Pclass']==3) & (test_data['Embarked']=='S')] = 0  # 3等舱S口岸女性

# 策略 E: V7 + 儿童 + 1/2等舱女性 + 3等舱女性低于阈值
pred_e = pred_a.copy()
pred_e[test_data['Age'] < 12] = 1
pred_e[(test_data['Sex']==1) & (test_data['Pclass']<=2)] = 1
# 3 等舱女性: 只有 Fare 高的存活
mask = (test_data['Sex']==1) & (test_data['Pclass']==3) & (test_data['Fare'] < 15)
pred_e[mask] = 0

# 策略 F: 性别基线 + 儿童 + 1等舱男性
pred_f = np.where(test_data['Sex']==1, 1, 0)  # 基线
pred_f[test_data['Age'] < 12] = 1  # 儿童
pred_f[(test_data['Sex']==0) & (test_data['Pclass']==1) & (test_data['Age'] < 50)] = 1  # 1等舱年轻男性
pred_f = pred_f.astype(int)

print('=== Prediction Distributions ===')
for name, pred in [('V7_orig', pred_a), ('V7+child', pred_b), ('V7+female12', pred_c),
                    ('V7+rules', pred_d), ('V7+fare', pred_e), ('Gender+child+1stmale', pred_f)]:
    print(f'{name:25s}: survived={pred.sum():3d}, died={418-pred.sum():3d}')

# 保存所有变体
for name, pred in [('v11a', pred_a), ('v11b', pred_b), ('v11c', pred_c),
                    ('v11d', pred_d), ('v11e', pred_e), ('v11f', pred_f)]:
    sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred})
    sub.to_csv(f'submission_{name}.csv', index=False)
    print(f'Saved submission_{name}.csv')
