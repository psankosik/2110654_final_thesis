import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DatasetLoader(object):

    @staticmethod
    def load_house_price():
        pass

    @staticmethod
    def load_iris():
        dataset_url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
        dataset = pd.read_csv(dataset_url)

        train, test = train_test_split(dataset, test_size=0.2, stratify=dataset["variety"], random_state=1234)

        x_train, y_train = train.values[:, :-1], train.values[:, -1]
        x_test, y_test = test.values[:, :-1], test.values[:, -1]

        # normalize
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        x_train = scaler.fit_transform(x_train)
        y_train = label_encoder.fit_transform(y_train)
        x_test = scaler.transform(x_test)
        y_test = label_encoder.transform(y_test)

        return x_train, y_train, x_test, y_test

    @staticmethod
    def load_titanic():
        dataset_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        dataset = pd.read_csv(dataset_url)

        dataset = pd.concat([
            dataset,
            pd.get_dummies(dataset.Sex, prefix='Sex'), 
            pd.get_dummies(dataset.Embarked, prefix='Embarked')
            ])
        dataset = dataset.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

        # TODO: Deal with missing value
        train = dataset.iloc[:-100, 1:]
        test = dataset.iloc[-100:, 1:]

        x_train, y_train = train.iloc[:, 1:].values, train.iloc[:, 0].values
        x_test, y_test = test.iloc[:, 1:].values, test.iloc[:, 0].values

        # normalize
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        x_train = scaler.fit_transform(x_train)
        y_train = label_encoder.fit_transform(y_train)
        x_test = scaler.transform(x_test)
        y_test = label_encoder.transform(y_test)
    
        return x_train, y_train, x_test, y_test