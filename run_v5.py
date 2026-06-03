"""
Titanic V5 - 基于 V2 特征集 + XGBoost/LightGBM 融合
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
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
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

# V2 相同的特征
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

# 调参 XGBoost
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8]
}
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', reg_alpha=0.1, reg_lambda=1.0),
    xgb_params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
)
xgb_grid.fit(X, y)
print(f'XGB Best: {xgb_grid.best_score_:.4f} | {xgb_grid.best_params_}')

# 调参 LightGBM
lgbm_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8]
}
lgbm_grid = GridSearchCV(
    LGBMClassifier(random_state=42, verbose=-1, reg_alpha=0.1, reg_lambda=1.0),
    lgbm_params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
)
lgbm_grid.fit(X, y)
print(f'LGBM Best: {lgbm_grid.best_score_:.4f} | {lgbm_grid.best_params_}')

# 其他模型 (V2 的参数)
rf = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_split=6, min_samples_leaf=3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, min_samples_split=6, min_samples_leaf=3, random_state=42)
svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, C=5, kernel='rbf', random_state=42))])
lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, C=1, random_state=42))])
knn = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))])

for name, m in [('RF', rf), ('GB', gb), ('SVM', svm), ('LR', lr), ('KNN', knn)]:
    s = cross_val_score(m, X, y, cv=cv, scoring='accuracy')
    print(f'{name:5s}: {s.mean():.4f}')

# 大融合 Voting
print('\n=== Ensemble ===')
big_voting = VotingClassifier(
    estimators=[
        ('xgb', xgb_grid.best_estimator_),
        ('lgbm', lgbm_grid.best_estimator_),
        ('rf', rf),
        ('gb', gb),
        ('svm', svm),
        ('lr', lr)
    ],
    voting='soft'
)
bv_scores = cross_val_score(big_voting, X, y, cv=cv, scoring='accuracy')
print(f'Big Voting: {bv_scores.mean():.4f} (+/- {bv_scores.std():.4f})')

# Stacking
big_stacking = StackingClassifier(
    estimators=[
        ('xgb', xgb_grid.best_estimator_),
        ('lgbm', lgbm_grid.best_estimator_),
        ('rf', rf),
        ('gb', gb),
        ('svm', svm)
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)
bs_scores = cross_val_score(big_stacking, X, y, cv=cv, scoring='accuracy')
print(f'Big Stacking: {bs_scores.mean():.4f} (+/- {bs_scores.std():.4f})')

# 选择最佳
candidates = {
    'XGB': (xgb_grid.best_score_, xgb_grid.best_estimator_),
    'LGBM': (lgbm_grid.best_score_, lgbm_grid.best_estimator_),
    'BigVoting': (bv_scores.mean(), big_voting),
    'BigStacking': (bs_scores.mean(), big_stacking),
}
best_name = max(candidates, key=lambda k: candidates[k][0])
best_score, best_model = candidates[best_name]
print(f'\nBest: {best_name} = {best_score:.4f}')

best_model.fit(X, y)
pred = best_model.predict(Xt)
sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred.astype(int)})
sub.to_csv('submission_v5.csv', index=False)
print(f'Submission saved: {sub.shape}')
