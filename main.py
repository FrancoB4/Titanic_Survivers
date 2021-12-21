from datetime import datetime
from pandas import DataFrame
from auxiliar_functions import one_preprocessing, read_csv
from sklearn.ensemble import RandomForestClassifier


def run():
    train_data, test_data = read_csv('./data/train.csv'), read_csv('./data/test.csv')

    x_train, y_train = one_preprocessing(train_data)
    x_test = one_preprocessing(test_data)

    model = RandomForestClassifier(n_estimators=150, max_depth=6).fit(x_train, y_train)

    predictions = model.predict(x_test)

    predictions_df = DataFrame(predictions, index=test_data['PassengerId'],
                                    columns=['Survived'])

    predictions_df.to_csv(f'./results/{datetime.now()}.csv')

if __name__ == '__main__':
    run()