"""
Titanic V8 - 错误分析 + 家庭存活分析 + 多种子平均
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

    # 新特征: 家庭存活率估计 (用同票号的其他乘客)
    # 在训练集上计算同 Ticket 的存活率
    if 'Survived' in df.columns:
        ticket_surv = df.groupby('Ticket')['Survived'].agg(['mean','count'])
        ticket_surv.columns = ['TicketSurvRate', 'TicketSurvCount']
        df = df.merge(ticket_surv, on='Ticket', how='left')
    else:
        df['TicketSurvRate'] = np.nan
        df['TicketSurvCount'] = np.nan

    # 同姓氏家庭存活率
    df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0])
    if 'Survived' in df.columns:
        surname_surv = df.groupby('Surname')['Survived'].agg(['mean','count'])
        surname_surv.columns = ['SurnameSurvRate', 'SurnameSurvCount']
        df = df.merge(surname_surv, on='Surname', how='left')
    else:
        df['SurnameSurvRate'] = np.nan
        df['SurnameSurvCount'] = np.nan

    # 编码
    for col in ['Embarked', 'Title', 'FamilyCat', 'CabinDeck', 'Sex*Pclass', 'Title*Pclass']:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        df = pd.concat([df, dummies], axis=1)
    return df

full_fe = engineer(full)

# 填充家庭存活率 (用全局平均)
full_fe['TicketSurvRate'] = full_fe['TicketSurvRate'].fillna(full_fe['TicketSurvRate'].median())
full_fe['TicketSurvCount'] = full_fe['TicketSurvCount'].fillna(0)
full_fe['SurnameSurvRate'] = full_fe['SurnameSurvRate'].fillna(full_fe['SurnameSurvRate'].median())
full_fe['SurnameSurvCount'] = full_fe['SurnameSurvCount'].fillna(0)

feature_cols = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'FamilySize', 'IsAlone', 'HasCabin', 'CabinNum', 'CabinCount',
    'IsChild', 'IsElderly', 'Age*Class', 'FarePerPerson', 'FareLog', 'FareBin',
    'TicketGroupSize', 'NameLen',
    'TicketSurvRate', 'TicketSurvCount', 'SurnameSurvRate', 'SurnameSurvCount'
]
feature_cols += [c for c in full_fe.columns if c.startswith(('Embarked_','Title_','FamilyCat_','CabinDeck_','Sex*Pclass_','Title*Pclass_'))]

train_fe = full_fe[full_fe['is_train']==1]
test_fe = full_fe[full_fe['is_train']==0]

X = train_fe[feature_cols].fillna(0)
y = train_fe['Survived'].astype(int)
Xt = test_fe[feature_cols].fillna(0)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# V7 的最佳模型组合
rf = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_split=6, min_samples_leaf=3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, min_samples_split=6, min_samples_leaf=3, random_state=42)
svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, C=5, kernel='rbf', random_state=42))])
lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, C=1, random_state=42))])
knn = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))])

v7 = VotingClassifier(
    estimators=[('rf',rf),('gb',gb),('svm',svm),('lr',lr),('knn',knn)],
    voting='soft'
)

s = cross_val_score(v7, X, y, cv=cv, scoring='accuracy')
print(f'V7+family CV: {s.mean():.4f} (+/- {s.std():.4f})')

# 多种子平均
print('\n=== Multi-seed averaging ===')
all_preds = []
for seed in range(50):
    cv_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    model = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_split=6, min_samples_leaf=3, random_state=seed)),
            ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, min_samples_split=6, min_samples_leaf=3, random_state=seed)),
            ('svm', Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, C=5, kernel='rbf', random_state=seed))])),
            ('lr', Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, C=1, random_state=seed))])),
            ('knn', Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))])),
        ],
        voting='soft'
    )
    model.fit(X, y)
    probs = model.predict_proba(Xt)[:, 1]
    all_preds.append(probs)

# 平均概率后取阈值
avg_probs = np.mean(all_preds, axis=0)

# 尝试不同阈值
print('\n=== Threshold search ===')
for thresh in [0.45, 0.48, 0.50, 0.52, 0.55]:
    preds = (avg_probs >= thresh).astype(int)
    print(f'Thresh {thresh}: {preds.sum()} survived / {len(preds)-preds.sum()} died')

# 用 0.5 阈值
final_preds = (avg_probs >= 0.5).astype(int)
sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': final_preds})
sub.to_csv('submission_v8.csv', index=False)
print(f'\nSubmission saved: {sub.shape}')
print(f'Survived: {final_preds.sum()}, Died: {len(final_preds)-final_preds.sum()}')
