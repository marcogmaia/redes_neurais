import pandas as pd
from pandas.io.parsers.readers import validate_integer

# TODO Remover a coluna dos índices


def data_prep_filter_types():
    df = pd.read_csv('TRNcod.csv')
    df1 = df.loc[df['IND_BOM_1_1'] == 1]
    df2 = df.loc[df['IND_BOM_1_2'] == 1]
    df1.to_csv('./class1.csv')
    df2.to_csv('./class2.csv')


def data_prep_random_sample():
    df1 = pd.read_csv('./class1.csv')
    df2 = pd.read_csv('./class2.csv')
    #
    # df1_50 = df1.sample(frac=0.5)
    # df11 = df1.drop(df1_50.index)
    # df1_25_1 = df11.sample(frac=0.5)
    # df12 = df11.drop(df1_25_1.index)
    # df1_50.to_csv('./class1_50.csv')
    # df1_25_1.to_csv('./class1_25_1.csv')
    # df12.to_csv('./class1_25_2.csv')

    df2 = pd.concat([df2, df2[:(len(df1) - len(df2))]], ignore_index=True)
    print(len(df2))

    df2_50 = df2.sample(frac=0.5)

    print('50% ', len(df2_50))
    df2.drop(df2_50.index, inplace=True)
    print(len(df2))
    df2_25_1 = df2.sample(frac=0.5)
    df2 = df2.drop(df2_25_1.index)
    print(len(df2))
    df2_25_2 = df2

    df2_50.to_csv('./class2_50.csv')
    df2_25_1.to_csv('./class2_25_1.csv')
    df2_25_2.to_csv('./class2_25_2.csv')

    print(f'1 linhas: {len(df1)}')
    print(f'2 linhas: {len(df2)}')

    # print(f'df1_50 linhas: {len(df1_50)}')
    # print(f'df1_25 linhas: {len(df1_25_1)}')


def data_prep():
    c1_50 = pd.read_csv('./data/class1_50.csv')
    c2_50 = pd.read_csv('./data/class2_50.csv')
    training_df = pd.concat([c1_50, c2_50], ignore_index=True)
    training_df.sample(frac=1.0).reset_index(drop=True)
    training_df.drop(['INDEX'], axis=1, inplace=True)
    training_df.to_csv('./traning_data.csv')

    c1_25_1 = pd.read_csv('./data/class1_25_1.csv')
    c2_25_1 = pd.read_csv('./data/class2_25_1.csv')
    validation_df = pd.concat([c1_25_1, c2_25_1], ignore_index=True)
    validation_df.sample(frac=1.0).reset_index(drop=True)
    validation_df.drop(['INDEX'], axis=1, inplace=True)
    validation_df.to_csv('./validation_data.csv')

    c1_25_2 = pd.read_csv('./data/class1_25_2.csv')
    c2_25_2 = pd.read_csv('./data/class2_25_2.csv')
    test_data = pd.concat([c1_25_2, c2_25_2], ignore_index=True)
    test_data.sample(frac=1.0).reset_index(drop=True)
    test_data.drop(['INDEX'], axis=1, inplace=True)
    test_data.to_csv('./test_data.csv')


# def main():
#     data_prep_random_sample()

if __name__ == '__main__':
    # main()
    data_prep()
    pass
