"""
Titanic V3 - 使用调参后的 GB 单模型 + 更多特征
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# 加载数据
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

    # 票号
    ticket_counts = df['Ticket'].value_counts()
    df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)

    # Cabin
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['CabinDeck'] = df['Cabin'].str[0].fillna('U')

    # 年龄 (按多维分组填充)
    for grp in [['Title','Pclass','Sex'], ['Title','Pclass']]:
        df['Age'] = df['Age'].fillna(df.groupby(grp)['Age'].transform('median'))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['IsChild'] = (df['Age'] < 12).astype(int)

    # 票价
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FarePerPerson'] = df['Fare'] / df['TicketGroupSize']
    df['FareLog'] = np.log1p(df['Fare'])
    df['FareBin'] = pd.qcut(df['Fare'], 5, labels=False, duplicates='drop')

    # 交互
    df['Age*Class'] = df['Age'] * df['Pclass']
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # 编码
    for col in ['Embarked', 'Title', 'CabinDeck']:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        df = pd.concat([df, dummies], axis=1)

    return df

full_fe = engineer(full)

features = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'FamilySize', 'IsAlone', 'HasCabin', 'IsChild',
    'FarePerPerson', 'FareLog', 'FareBin', 'Age*Class',
    'TicketGroupSize', 'FamilyCat'
]
features += [c for c in full_fe.columns if c.startswith(('Embarked_','Title_','CabinDeck_'))]

train_fe = full_fe[full_fe['is_train']==1]
test_fe = full_fe[full_fe['is_train']==0]

X = train_fe[features].fillna(0)
y = train_fe['Survived'].astype(int)
Xt = test_fe[features].fillna(0)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 最优 GB 参数
gb = GradientBoostingClassifier(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    min_samples_leaf=5, subsample=0.8,
    min_samples_split=4, random_state=42
)
scores = cross_val_score(gb, X, y, cv=cv, scoring='accuracy')
print(f'GB CV: {scores.mean():.4f} (+/- {scores.std():.4f})')

# 训练并预测
gb.fit(X, y)
pred = gb.predict(Xt)

sub = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred.astype(int)})
sub.to_csv('submission_v3.csv', index=False)
print(f'Submission saved: {sub.shape}')
