"""
Titanic V7 - 基于 V2 成功经验 + XGBoost/LightGBM 加入融合
V2 成功要素: 多维特征 + 多模型 Voting
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
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
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

# V2 完全相同的特征工程
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

# V2 的模型 (调参后)
rf = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_split=6, min_samples_leaf=3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, min_samples_split=6, min_samples_leaf=3, random_state=42)
svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, C=5, kernel='rbf', random_state=42))])
lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, C=1, random_state=42))])

# 新增 XGBoost/LightGBM (保守参数)
xgb = XGBClassifier(
    n_estimators=150, max_depth=3, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=3.0,
    random_state=42, eval_metric='logloss'
)
lgbm = LGBMClassifier(
    n_estimators=150, max_depth=3, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=3.0,
    random_state=42, verbose=-1
)

print('=== Individual CV ===')
all_models = {'RF': rf, 'GB': gb, 'SVM': svm, 'LR': lr, 'XGB': xgb, 'LGBM': lgbm}
for name, m in all_models.items():
    s = cross_val_score(m, X, y, cv=cv, scoring='accuracy')
    print(f'{name:5s}: {s.mean():.4f}')

# V2 Voting (5 models) vs 加入 XGB/LGBM 后的 Voting (7 models)
print('\n=== Ensemble ===')
v5 = VotingClassifier(
    estimators=[('rf',rf),('gb',gb),('svm',svm),('lr',lr),('knn',
        Pipeline([('scaler',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=5,weights='distance'))]))],
    voting='soft'
)
s5 = cross_val_score(v5, X, y, cv=cv, scoring='accuracy')
print(f'V2 Voting (5): {s5.mean():.4f}')

v7 = VotingClassifier(
    estimators=[('rf',rf),('gb',gb),('svm',svm),('lr',lr),('xgb',xgb),('lgbm',lgbm)],
    voting='soft'
)
s7 = cross_val_score(v7, X, y, cv=cv, scoring='accuracy')
print(f'V7 Voting (6): {s7.mean():.4f}')

# 选择最佳
if s7.mean() >= s5.mean():
    best = v7
    best_name = 'V7'
else:
    best = v5
    best_name = 'V2-style'

print(f'\nUsing: {best_name}')
best.fit(X, y)
pred = best.predict(Xt)
sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred.astype(int)})
sub.to_csv('submission_v7.csv', index=False)
print(f'Submission saved: {sub.shape}')
