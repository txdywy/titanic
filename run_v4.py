"""
Titanic V4 - XGBoost + LightGBM + 精选特征
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# 加载
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train['is_train'] = 1
test['is_train'] = 0
test['Survived'] = np.nan
full = pd.concat([train, test], sort=False).reset_index(drop=True)

def engineer(df):
    df = df.copy()

    # 称谓
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {'Mr':'Mr','Miss':'Miss','Mrs':'Mrs','Master':'Master',
                 'Dr':'Rare','Rev':'Rare','Col':'Rare','Major':'Rare',
                 'Mlle':'Miss','Ms':'Miss','Mme':'Mrs','Don':'Rare',
                 'Dona':'Rare','Lady':'Rare','Countess':'Rare',
                 'Jonkheer':'Rare','Sir':'Rare','Capt':'Rare'}
    df['Title'] = df['Title'].map(title_map).fillna('Rare')

    # 家庭
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FamilyCat'] = df['FamilySize'].map(lambda x: 0 if x==1 else 1 if x<=4 else 2)

    # 票号同行
    ticket_counts = df['Ticket'].value_counts()
    df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)

    # Cabin
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # 年龄
    for grp in [['Title','Pclass','Sex'], ['Title','Pclass']]:
        df['Age'] = df['Age'].fillna(df.groupby(grp)['Age'].transform('median'))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['IsChild'] = (df['Age'] < 12).astype(int)

    # 票价
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']
    df['FareLog'] = np.log1p(df['Fare'])

    # 交互
    df['Age*Class'] = df['Age'] * df['Pclass']
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # 编码
    for col in ['Embarked', 'Title']:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        df = pd.concat([df, dummies], axis=1)

    return df

full_fe = engineer(full)

# 精选核心特征 (避免过拟合)
features = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
    'FamilySize', 'IsAlone', 'HasCabin', 'IsChild',
    'FarePerPerson', 'FareLog', 'Age*Class',
    'TicketGroupSize', 'FamilyCat'
]
features += [c for c in full_fe.columns if c.startswith(('Embarked_','Title_'))]

train_fe = full_fe[full_fe['is_train']==1]
test_fe = full_fe[full_fe['is_train']==0]

X = train_fe[features].fillna(0)
y = train_fe['Survived'].astype(int)
Xt = test_fe[features].fillna(0)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 各模型
models = {
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1
    ),
    'RF': RandomForestClassifier(
        n_estimators=200, max_depth=7, min_samples_split=4,
        min_samples_leaf=2, random_state=42
    ),
    'GB': GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=42
    ),
    'LR': LogisticRegression(max_iter=1000, C=1.0, random_state=42)
}

print('=== Individual Models ===')
best_score = 0
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f'{name:12s}: {scores.mean():.4f} (+/- {scores.std():.4f})')
    if scores.mean() > best_score:
        best_score = scores.mean()

# Voting ensemble (top 3 models)
print('\n=== Ensemble ===')
voting = VotingClassifier(
    estimators=[(n, m) for n, m in models.items()],
    voting='soft'
)
v_scores = cross_val_score(voting, X, y, cv=cv, scoring='accuracy')
print(f'Voting (all): {v_scores.mean():.4f} (+/- {v_scores.std():.4f})')

# 用 voting 提交
voting.fit(X, y)
pred = voting.predict(Xt)
sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred.astype(int)})
sub.to_csv('submission_v4.csv', index=False)
print(f'\nSubmission saved: {sub.shape}')
