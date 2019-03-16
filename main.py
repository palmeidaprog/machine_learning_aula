import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
import warnings
from os import listdir
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def scores(clf_name, prediction, metodo, target_test, file, split_number, iteracao, output):
        with open(output, 'at') as out_file:
            line = f"\"{file} - {clf_name} - {metodo} - Split # {split_number} - Treino # {iteracao}\","
            line += f"{accuracy_score(target_test, prediction)},"
            line += f"{matthews_corrcoef(target_test, prediction)},"
            line += f"{f1_score(target_test, prediction,average='macro')},"
            line += f"{recall_score(target_test, prediction, average='macro')},"
            line += f"{precision_score(target_test, prediction, average='macro')}\n"
            out_file.writelines(line)
            #print(f"{classification_report(self.treated_data.target_test, self.prediction)}")
        pass

dir = 'datasets/'
output = 'output.csv'
names=[]

for file in listdir(dir):
    print(f"---{file}---")

    with open(dir+file, 'rt') as in_file:
        for line in in_file:
            if line.startswith("@inputs"):
                for word in line.split(" "):
                    if word != '@inputs':
                        names.append(word.replace('\n', '').replace(',', ''))
                names.append("classes")
            if line.startswith("@data"):
                break

    with open(output, 'wt') as out_file: 
        out_file.writelines('\"Descrição\",\"Acurácia\",\"F1-Score\",\"Recall\",\"Precisão,MCC\"\n')

    # Le arquivo CSV, para pandas dataframe
    data = pd.read_csv(dir+file, comment='@', names=names)

    # transforma dados categóricos em dados numéricos ()
    encoder = LabelEncoder()
    data = data.apply(encoder.fit_transform)

    # transforma o dataframe em uma matriz com os features (ft) e 
    # cria um vetor com os alvos/targets (tg), nome das classes 
    ultimaColuna = len(names) - 1
    ft = data.iloc[:, 0:ultimaColuna]
    tg = data.iloc[:, ultimaColuna]

    for i in range(5):
        ft_train, ft_test, tg_train, tg_test = train_test_split(ft, tg, \
            train_size=0.75)
        ft_train, ft_valid, tg_train, tg_valid = train_test_split(ft_train, tg_train, \
            train_size=0.9)

        s = StandardScaler()
        padr_ft_train = s.fit_transform(ft_train)
        padr_ft_test = s.transform(ft_test)
                
        n = Normalizer()
        norm_ft_train = s.fit_transform(ft_train)
        norm_ft_test = s.transform(ft_test)

        for j in range(30):
            percep = Perceptron(max_iter=100, random_state=0 ,eta0=0.1, n_jobs=-1)
            percep.fit(norm_ft_train, tg_train)
            norm_prediction = percep.predict(norm_ft_test)
            scores("Perceptron", norm_prediction, "Normalizado", tg_test, file, i, j, output)

            percep = Perceptron(max_iter=100, random_state=0 ,eta0=0.1, n_jobs=-1)
            percep.fit(padr_ft_train, tg_train)
            padr_prediction = percep.predict(padr_ft_test)
            scores("Perceptron", padr_prediction, "Padronizado", tg_test, file, i, j, output)

            dt = DecisionTreeClassifier(random_state=0)
            dt.fit(norm_ft_train, tg_train)
            norm_prediction = dt.predict(norm_ft_test)
            scores("Decision Tree", norm_prediction, "Normalizado", tg_test, file, i, j, output)

            dt = DecisionTreeClassifier(random_state=0)
            dt.fit(padr_ft_train, tg_train)
            padr_prediction = dt.predict(padr_ft_test)
            scores("Decision Tree", padr_prediction, "Padronizado", tg_test, file, i, j, output)

            # lda = LinearDiscriminantAnalysis()
            # features_r = lda.fit(padr_ft_train, tg_train).transform(padr_ft_train)
            # padr_prediction = lda.predict(padr_ft_test) 
            # scores("LDA", padr_prediction, "Padronizado", tg_test, file, i, j, output)

            # lda = LinearDiscriminantAnalysis()
            # features_r = lda.fit(norm_ft_train, tg_train).transform(norm_ft_train)
            # norm_prediction = lda.predict(norm_ft_test) 
            # scores("LDA", norm_prediction, "Normalizado", tg_test, file, i, j, output)

