"""
Titanic V10 - V7 最佳组合 + XGB/LGBM + Optuna 权重优化
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
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# V7 模型
rf = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_split=6, min_samples_leaf=3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, min_samples_split=6, min_samples_leaf=3, random_state=42)
svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, C=5, kernel='rbf', random_state=42))])
lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, C=1, random_state=42))])
knn = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))])

# 新增
xgb = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8,
                     reg_alpha=0.5, reg_lambda=3.0,
                     random_state=42, eval_metric='logloss')
lgbm = LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.1,
                       subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.5, reg_lambda=3.0,
                       random_state=42, verbose=-1)

# 测试不同组合
print('=== Voting Combos ===')
combos = [
    ('v7_5model', [('rf',rf),('gb',gb),('svm',svm),('lr',lr),('knn',knn)]),
    ('v7+xgb', [('rf',rf),('gb',gb),('svm',svm),('lr',lr),('knn',knn),('xgb',xgb)]),
    ('v7+lgbm', [('rf',rf),('gb',gb),('svm',svm),('lr',lr),('knn',knn),('lgbm',lgbm)]),
    ('v7+xgb+lgbm', [('rf',rf),('gb',gb),('svm',svm),('lr',lr),('knn',knn),('xgb',xgb),('lgbm',lgbm)]),
    ('svm+lr+knn+gb+xgb', [('svm',svm),('lr',lr),('knn',knn),('gb',gb),('xgb',xgb)]),
    ('svm+lr+knn+gb+lgbm', [('svm',svm),('lr',lr),('knn',knn),('gb',gb),('lgbm',lgbm)]),
]

best_score = 0
best_name = ''
best_estimators = []
for name, estimators in combos:
    vc = VotingClassifier(estimators=estimators, voting='soft')
    s = cross_val_score(vc, X, y, cv=cv, scoring='accuracy')
    print(f'{name:25s}: {s.mean():.4f} (+/- {s.std():.4f})')
    if s.mean() > best_score:
        best_score = s.mean()
        best_name = name
        best_estimators = estimators

print(f'\nBest: {best_name} = {best_score:.4f}')

# 用最佳组合多提交几个变体
best_vc = VotingClassifier(estimators=best_estimators, voting='soft')
best_vc.fit(X, y)

# 变体1: 最佳组合
pred1 = best_vc.predict(Xt)
sub1 = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred1.astype(int)})
sub1.to_csv('submission_v10a.csv', index=False)

# 变体2: V7 原版 (我们的最佳公榜分数)
v7 = VotingClassifier(estimators=[('rf',rf),('gb',gb),('svm',svm),('lr',lr),('knn',knn)], voting='soft')
v7.fit(X, y)
pred2 = v7.predict(Xt)
sub2 = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred2.astype(int)})
sub2.to_csv('submission_v10b.csv', index=False)

# 变体3: 加入 XGBoost
v7x = VotingClassifier(estimators=[('rf',rf),('gb',gb),('svm',svm),('lr',lr),('knn',knn),('xgb',xgb)], voting='soft')
v7x.fit(X, y)
pred3 = v7x.predict(Xt)
sub3 = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred3.astype(int)})
sub3.to_csv('submission_v10c.csv', index=False)

print(f'All submissions saved')
