"""
Titanic V6 - 领域知识驱动 + 保守特征 + 强正则化
策略: "Women and children first" + 社会阶层
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
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

    # 核心信号
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})

    # 称谓 (关键特征)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {'Mr':'Mr','Miss':'Miss','Mrs':'Mrs','Master':'Master',
                 'Dr':'Rare','Rev':'Rare','Col':'Rare','Major':'Rare',
                 'Mlle':'Miss','Ms':'Miss','Mme':'Mrs','Don':'Rare',
                 'Dona':'Rare','Lady':'Rare','Countess':'Rare',
                 'Jonkheer':'Rare','Sir':'Rare','Capt':'Rare'}
    df['Title'] = df['Title'].map(title_map).fillna('Rare')

    # 年龄 (保守填充)
    df['Age'] = df['Age'].fillna(df.groupby(['Title','Pclass'])['Age'].transform('median'))
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # 家庭
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 票价
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # 登船
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # 仅使用最稳定的特征
    # 核心: Sex, Pclass (最强信号)
    # 次级: Title (捕获年龄+性别+社会地位), FamilySize, Fare
    # 保护: Age 粗粒度使用

    df['IsChild'] = (df['Age'] < 12).astype(int)
    df['IsMother'] = ((df['Sex']==1) & (df['Age']>18) & (df['Parch']>0) & (df['Title']!='Miss')).astype(int)

    # 编码
    for col in ['Embarked', 'Title']:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        df = pd.concat([df, dummies], axis=1)

    return df

full_fe = engineer(full)

# 极简特征集 - 只用最强信号
features = [
    'Sex', 'Pclass', 'Age', 'IsChild', 'IsMother',
    'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Fare'
]
features += [c for c in full_fe.columns if c.startswith(('Embarked_','Title_'))]

train_fe = full_fe[full_fe['is_train']==1]
test_fe = full_fe[full_fe['is_train']==0]

X = train_fe[features].fillna(0)
y = train_fe['Survived'].astype(int)
Xt = test_fe[features].fillna(0)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 保守模型 (高正则化)
models = {
    'XGB_conservative': XGBClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.1,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=5.0,
        random_state=42, eval_metric='logloss'
    ),
    'GB_conservative': GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.1,
        min_samples_leaf=10, subsample=0.7,
        random_state=42
    ),
    'RF_conservative': RandomForestClassifier(
        n_estimators=200, max_depth=5,
        min_samples_split=10, min_samples_leaf=5,
        random_state=42
    ),
    'LR': LogisticRegression(max_iter=1000, C=0.5, random_state=42),
}

print('=== Conservative Models ===')
for name, m in models.items():
    s = cross_val_score(m, X, y, cv=cv, scoring='accuracy')
    print(f'{name:20s}: {s.mean():.4f} (+/- {s.std():.4f})')

# 多个 Voting 组合
combos = [
    ('xgb+gb+lr', ['XGB_conservative', 'GB_conservative', 'LR']),
    ('xgb+gb+rf+lr', ['XGB_conservative', 'GB_conservative', 'RF_conservative', 'LR']),
    ('all', list(models.keys())),
]

print('\n=== Voting Combos ===')
best_combo_score = 0
best_combo_name = ''
for combo_name, model_names in combos:
    ests = [(n, models[n]) for n in model_names]
    vc = VotingClassifier(estimators=ests, voting='soft')
    s = cross_val_score(vc, X, y, cv=cv, scoring='accuracy')
    print(f'{combo_name:20s}: {s.mean():.4f} (+/- {s.std():.4f})')
    if s.mean() > best_combo_score:
        best_combo_score = s.mean()
        best_combo_name = combo_name

# 用最佳组合提交
print(f'\nBest combo: {best_combo_name} = {best_combo_score:.4f}')
best_mn = [mn for cn, mn in combos if cn == best_combo_name][0]
ests = [(n, models[n]) for n in best_mn]
best_ensemble = VotingClassifier(estimators=ests, voting='soft')
best_ensemble.fit(X, y)
pred = best_ensemble.predict(Xt)

sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred.astype(int)})
sub.to_csv('submission_v6.csv', index=False)
print(f'Submission saved: {sub.shape}')
