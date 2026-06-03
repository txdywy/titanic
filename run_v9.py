"""
Titanic V9 - Optuna 超参优化 + Target Encoding + 精细特征
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier, StackingClassifier)
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
    # 新交互
    df['AgeBin'] = pd.cut(df['Age'], bins=[0,5,12,18,35,60,100], labels=[0,1,2,3,4,5])
    df['Fare*Pclass'] = df['Fare'] * df['Pclass']
    df['FamilySurvival'] = df['SibSp'] * df['Sex']  # 女性有兄弟姐妹
    for col in ['Embarked', 'Title', 'FamilyCat', 'CabinDeck', 'Sex*Pclass', 'Title*Pclass']:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        df = pd.concat([df, dummies], axis=1)
    return df

full_fe = engineer(full)

feature_cols = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'FamilySize', 'IsAlone', 'HasCabin', 'CabinNum', 'CabinCount',
    'IsChild', 'IsElderly', 'Age*Class', 'FarePerPerson', 'FareLog', 'FareBin',
    'TicketGroupSize', 'NameLen', 'AgeBin', 'Fare*Pclass', 'FamilySurvival'
]
feature_cols += [c for c in full_fe.columns if c.startswith(('Embarked_','Title_','FamilyCat_','CabinDeck_','Sex*Pclass_','Title*Pclass_'))]

train_fe = full_fe[full_fe['is_train']==1]
test_fe = full_fe[full_fe['is_train']==0]
X = train_fe[feature_cols].fillna(0).values
y = train_fe['Survived'].astype(int).values
Xt = test_fe[feature_cols].fillna(0).values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Optuna 优化 XGBoost
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(xgb_objective, n_trials=100, show_progress_bar=False)
print(f'XGB Optuna: {study_xgb.best_value:.4f}')

# Optuna 优化 LightGBM
def lgbm_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'verbose': -1
    }
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(lgbm_objective, n_trials=100, show_progress_bar=False)
print(f'LGBM Optuna: {study_lgbm.best_value:.4f}')

# 最优模型
best_xgb = XGBClassifier(**study_xgb.best_params, random_state=42, eval_metric='logloss')
best_lgbm = LGBMClassifier(**study_lgbm.best_params, random_state=42, verbose=-1)

# 传统模型
rf = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_split=6, min_samples_leaf=3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, min_samples_split=6, min_samples_leaf=3, random_state=42)
svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, C=5, kernel='rbf', random_state=42))])
lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, C=1, random_state=42))])
knn = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))])

# 测试各种组合
print('\n=== Combos ===')
combos = [
    ('v7_original', [('rf',rf),('gb',gb),('svm',svm),('lr',lr),('knn',knn)]),
    ('v7+xgb+lgbm', [('rf',rf),('gb',gb),('svm',svm),('lr',lr),('knn',knn),('xgb',best_xgb),('lgbm',best_lgbm)]),
    ('top4_tree', [('rf',rf),('gb',gb),('xgb',best_xgb),('lgbm',best_lgbm)]),
    ('top3', [('gb',gb),('xgb',best_xgb),('lgbm',best_lgbm)]),
    ('svm+lr+knn+gb', [('svm',svm),('lr',lr),('knn',knn),('gb',gb)]),
]

best_score = 0
best_combo = ''
for name, estimators in combos:
    vc = VotingClassifier(estimators=estimators, voting='soft')
    s = cross_val_score(vc, X, y, cv=cv, scoring='accuracy')
    print(f'{name:20s}: {s.mean():.4f} (+/- {s.std():.4f})')
    if s.mean() > best_score:
        best_score = s.mean()
        best_combo = name

# Stacking
print('\n=== Stacking ===')
stack = StackingClassifier(
    estimators=[('rf',rf),('gb',gb),('xgb',best_xgb),('lgbm',best_lgbm),('svm',svm)],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)
ss = cross_val_score(stack, X, y, cv=cv, scoring='accuracy')
print(f'Stacking: {ss.mean():.4f} (+/- {ss.std():.4f})')

# 选择最佳
print(f'\nBest combo: {best_combo} = {best_score:.4f}')
if ss.mean() > best_score:
    print(f'Stacking better: {ss.mean():.4f}')
    best_model = stack
else:
    best_model = VotingClassifier(estimators=[e for n, e in combos if n == best_combo][0], voting='soft')

best_model.fit(X, y)
pred = best_model.predict(Xt)
sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred.astype(int)})
sub.to_csv('submission_v9.csv', index=False)
print(f'Submission saved: {sub.shape}')
