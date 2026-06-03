"""
Titanic V12 - 加权 Voting + Optuna 权重优化
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

# 基础模型
rf = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_split=6, min_samples_leaf=3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, min_samples_split=6, min_samples_leaf=3, random_state=42)
svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC(probability=True, C=5, kernel='rbf', random_state=42))])
lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, C=1, random_state=42))])
knn = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))])

# Optuna 优化权重
def weight_objective(trial):
    w_rf = trial.suggest_float('w_rf', 0.5, 3.0)
    w_gb = trial.suggest_float('w_gb', 0.5, 3.0)
    w_svm = trial.suggest_float('w_svm', 0.5, 3.0)
    w_lr = trial.suggest_float('w_lr', 0.5, 3.0)
    w_knn = trial.suggest_float('w_knn', 0.5, 3.0)

    vc = VotingClassifier(
        estimators=[('rf',rf),('gb',gb),('svm',svm),('lr',lr),('knn',knn)],
        voting='soft',
        weights=[w_rf, w_gb, w_svm, w_lr, w_knn]
    )
    scores = cross_val_score(vc, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(weight_objective, n_trials=200, show_progress_bar=False)
print(f'Best weighted Voting CV: {study.best_value:.4f}')
print(f'Best weights: {study.best_params}')

# 用最优权重
best_weights = [study.best_params[f'w_{n}'] for n in ['rf','gb','svm','lr','knn']]
weighted_vc = VotingClassifier(
    estimators=[('rf',rf),('gb',gb),('svm',svm),('lr',lr),('knn',knn)],
    voting='soft',
    weights=best_weights
)
weighted_vc.fit(X, y)

# Stacking
stack = StackingClassifier(
    estimators=[('rf',rf),('gb',gb),('svm',svm),('lr',lr),('knn',knn)],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5
)
ss = cross_val_score(stack, X, y, cv=cv, scoring='accuracy')
print(f'Stacking CV: {ss.mean():.4f}')
stack.fit(X, y)

# 提交
for name, model in [('weighted_voting', weighted_vc), ('stacking', stack)]:
    pred = model.predict(Xt)
    sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred.astype(int)})
    sub.to_csv(f'submission_{name}.csv', index=False)
    print(f'Saved submission_{name}.csv')
