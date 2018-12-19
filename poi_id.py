#!/usr/bin/python

import sys
import pickle
#sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

import numpy as np
from matplotlib import pyplot as plt

features_list = ['poi','salary','bonus','shared_receipt_with_poi','from_this_person_to_poi','from_poi_to_this_person','expenses'] # You will need to use more features

### Carregar o dicionário que contém o conjunto de dados
import os
os.chdir(r'C:\udacity_optum\02 - Fundamentos de Data Science II\Projeto_04\Projeto_Feito')
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

def explore_data():
    
    #Tamanho do dataset 
    print 'Existem %d pessoas no dataset' % len(df)

    #nome das colunas
    print df.columns

    #tipos dos dados
    print df.info()

    #dados ausentes
    print pd.isnull(df).sum()

    #Visao estatistica do dataset completo
    print df.describe()

    #Visao estatistica dos POI
    print df[df.poi.isin([True])].describe()

    #Visao estatistica dos nao POI
    print df[df.poi.isin([False])].describe()

    #Numero de atributos
    print len(df.columns) 

    #Numero de POIs
    print len(df[df.poi.isin([True])])
    #Numero de nao POIs
    print len(df[df.poi.isin([False])])


#########################################################################################
#                               Explorar os dados                                        #
#########################################################################################
explore_data()


############    Task 2: Remover outliers  ################

    
data_temp = featureFormat(data_dict, features_list, sort_keys = True, remove_NaN = True)
salary = data_temp[:,1]
bonus = data_temp[:,2]
shared_receipt_with_poi = data_temp[:,3]
from_this_person_to_poi = data_temp[:,4]
from_poi_to_this_person = data_temp[:,5]
expenses = data_temp[:,6]
# Plotagem de gráficos de dispersão para identificar outliers

new_features_list = features_list[1:]
fig_count=0
for i in range(len(new_features_list)-2):
    i = i+2
    for j in range(i+1,len(new_features_list)):
        if i!=j:
            fig_count=fig_count+1
            plt.figure(fig_count)
            plt.plot(vars()[new_features_list[i]],vars()[new_features_list[j]],'.')
            plt.xlabel(new_features_list[i])
            plt.ylabel(new_features_list[j])

# Plotagem de bônus x salário
plt.figure()
plt.scatter(salary,bonus)
plt.xlabel('Salary')
plt.ylabel('bonus')


# Removendo o outlier ('TOTAL')
data_dict.pop('TOTAL',0)

data_new = featureFormat(data_dict, features_list, remove_NaN=True)  # datadado sem outlier
salary_new=data_new[:,1]
bonus_new=data_new[:,2]
expenses_new=data_new[:,6]

# Plotagem salary vs bonus depois de remover o outlier 'TOTAL'
plt.figure()
plt.scatter(salary_new,bonus_new)
plt.xlabel('Salary')
plt.ylabel('bonus')


    
###########    Task 3: Criar novas feature(s)  ############


### Armazenar em my_dataset para facilitar a exportação abaixo.
my_dataset = data_dict

### Extrair features e labels do conjunto de dados para os testes locais
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



data_pts = len(data)
poi_pts = sum(data[:,0])
print('Número de pontos de dados: %i' % data_pts)
print('Número de POIs : %i' % poi_pts)
print('Números de features usadas : %i' % (len(features_list)-1))

feature_count=0
for f in features_list:
    print('Números de NaN em %s = %i' % (f,sum(np.isnan(data[:,feature_count]))))
    feature_count = feature_count+1


# Remover NaN dos dados
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN=True)

# Testar a feature 

total_tofrom_poi = data_new[:,4] + data_new[:,5]
total_tofrom_poi = total_tofrom_poi.reshape(-1,1)
data = np.append(data,total_tofrom_poi,axis=1)

labels, features = targetFeatureSplit(data)


# Implementar valores de escala em algumas features é muito grande e varia em um intervalo maior
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
features = scaler.fit_transform(features)

###############k

def select_best_features(n_features):
    data = featureFormat(data_dict, features_list, remove_all_zeroes = False, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    selector = SelectKBest(k = n_features)  
    selector.fit_transform(features, labels)
    selected_indices = selector.get_support(indices=True)
    final_features = []
    for indice in selected_indices:
        #print 'feature -> {} with score -> {}'.format(features_list[indice + 1], selector.scores_[indice])
        final_features.append(features_list[indice + 1])
    return final_features

def selectKBest_f1_scores(clf, dataset, n_kbest_features, folds = 1000):
     graficoX = []
    graficoY = []
    for k in range(2, n_kbest_features):
        features_selected = select_best_features(k)
        features_selected.insert(0, "poi")
        data = featureFormat(dataset, features_selected, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
        true_negatives = 0
        false_negatives = 0
        true_positives = 0
        false_positives = 0
        for train_idx, test_idx in cv: 
            features_train = []
            features_test  = []
            labels_train   = []
            labels_test    = []
            for ii in train_idx:
                features_train.append( features[ii] )
                labels_train.append( labels[ii] )
            for jj in test_idx:
                features_test.append( features[jj] )
                labels_test.append( labels[jj] )

            clf.fit(features_train, labels_train)
            predictions = clf.predict(features_test)
            for prediction, truth in zip(predictions, labels_test):
                if prediction == 0 and truth == 0:
                    true_negatives += 1
                elif prediction == 0 and truth == 1:
                    false_negatives += 1
                elif prediction == 1 and truth == 0:
                    false_positives += 1
                elif prediction == 1 and truth == 1:
                    true_positives += 1
                else:
                    print ("Warning: Found a predicted label not == 0 or 1.")
                    print ("All predictions should take value 0 or 1.")
                    print ("Evaluating performance for processed predictions:")
                    break
        try:
            f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
            graficoY.append(f1)
            graficoX.append(k)
        except:
            print ("Got a divide by zero when trying out:", clf)
            print ("Precision or recall may be undefined due to a lack of true positive predicitons.")
    return  graficoX, graficoY


########### Gráfico ###################
x, y = selectKBest_f1_scores(GaussianNB(), data_dict, 20)
plt.figure()
plt.xlabel("Numero de features selecionadas")
plt.ylabel("Valor de F1-Score")
plt.plot(x, y)
plt.savefig('featureSelection.png', transparent=True)
plt.show()
##############k

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
#from sklearn import cross_validation
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)



############ Testar o algoritmo Gaussian NB ##############


clf_NB = GaussianNB()
clf_NB.fit(features_train,labels_train)
pred_NB = clf_NB.predict(features_test)
acc_NB = accuracy_score(pred_NB,labels_test)
print('Acurácia_NB_COM_NOVAS_FEATURES: %0.3f' % acc_NB)

precision_NB = precision_score(labels_test,pred_NB)
recall_NB = recall_score(labels_test,pred_NB)
print('Precisão_NB_COM_NOVAS_FEATURES = %0.3f' % precision_NB)
print('Recall_NB_COM_NOVAS_FEATURES = %0.3f' % recall_NB)



####### Testar o Classificador de Árvore de Decisão (DT) ###################


param_grid_DT1 = {
             'min_samples_split': [4,8,2,6],
              'min_samples_leaf': [3,2,1,4],'max_depth':[2,3,4]}
clf_DT=GridSearchCV(tree.DecisionTreeClassifier(random_state = 0), param_grid_DT1)
clf_DT.fit(features_train,labels_train)
print ("Melhores estimadores para DT_COM_NOVAS_FEATURES:")
print (clf_DT.best_estimator_)

pred_DT=clf_DT.predict(features_test)
acc_DT=accuracy_score(pred_DT, labels_test)
print('Acurácia_DT_COM_NOVAS_FEATURES = %0.3f' % acc_DT)

precision_DT = precision_score(labels_test,pred_DT)
recall_DT = recall_score(labels_test,pred_DT)
print('Precisão_DT_COM_NOVAS_FEATURES = %0.3f' % precision_DT)
print('Recall_DT_COM_NOVAS_FEATURES = %0.3f' % recall_DT)





###### Testar o algoritmo SVM ##########################

from sklearn.svm import SVC

param_grid_svm = {
             'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.001, 0.0005, 0.0001, 0.005, 0.01, 0.1],
              }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf_svm = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid_svm)
clf_svm = clf_svm.fit(features_train, labels_train)
print ("Melhores estimadores para SVM_COM_NOVAS_FEATURES:")
print (clf_svm.best_estimator_)

pred_svm = clf_svm.predict(features_test)
acc_svm = accuracy_score(pred_svm, labels_test)
print('Acurácia_SVM_COM_NOVAS_FEATURES = %0.3f' % acc_svm)


precision_svm = precision_score(labels_test,pred_svm)
recall_svm = recall_score(labels_test,pred_svm)
print('Precisão_SVM_COM_NOVAS_FEATURES = %0.3f' % precision_svm)
print('Recall_SVM_COM_NOVAS_FEATURES = %0.3f' % recall_svm)


###########################################


### Obter os valores de importância para cada feature usando DT

important_features=clf_DT.best_estimator_.feature_importances_
new_features_list = new_features_list + ['total_tofrom_poi']
idx_features = 0
for idx in important_features:
    print (idx)
    if (idx > 0.02):
        print('Feature = %s' % new_features_list[idx_features], 'Importância = %0.5f' % idx )      # imprimir features com importância > 0.2
    idx_features = idx_features + 1
        
max_value=max(important_features)               # valor  máx. da importância
index_max = np.argmax(important_features)       # índice do valor  máx. da importância

print('Valor máx. da importância da feature = %0.7f' % max_value, 'Índice da feature mais importante = %i' % index_max)
print('Feature mais importante = %s' % new_features_list[index_max])




############  Análise Final ###########

# NÃO incluindo a nova feature

# Classificador usado = Árvores de Decisão (DT)

my_dataset = data_dict

# Selecionar os três recursos mais importantes (por importâncias de features)
features_list = ['poi','shared_receipt_with_poi','from_this_person_to_poi','expenses']

data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN=True)
labels, features = targetFeatureSplit(data)

# Implementar valores de escala de feature, em algumas features é muito grande e varia em um intervalo maior
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
features_scaled = scaler.fit_transform(features)


features_train, features_test, labels_train, labels_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)


param_grid_DT = {
             'min_samples_split': [2,4,8,6],
              'min_samples_leaf': [1,2,3,4]}
clf=GridSearchCV(tree.DecisionTreeClassifier(random_state = 42), param_grid_DT)
#clf=tree.DecisionTreeClassifier(random_state = 42, min_samples_leaf=2, min_samples_split=2)
clf.fit(features_train,labels_train)
print ("Melhores estimadores para DT SEM NOVA FEATURES:")
print (clf.best_estimator_)

pred=clf.predict(features_test)
acc=accuracy_score(pred, labels_test)
print('Acurácia_DT SEM NOVA FEATURES= %0.3f' % acc)

precision = precision_score(labels_test,pred)
recall = recall_score(labels_test,pred)
print('Precisão_DT SEM NOVA FEATURES = %0.3f' % precision)
print('Recall_DT SEM NOVA FEATURES = %0.3f' % recall)


#### Outro algoritmo testado na análise final

#####################################


### Naive Bayes

clf_NB = GaussianNB()
clf_NB.fit(features_train,labels_train)
pred_NB = clf_NB.predict(features_test)
acc_NB = accuracy_score(pred_NB,labels_test)
print('Acurácia_NB SEM NOVA FEATURES: %0.3f' % acc_NB)

precision_NB = precision_score(labels_test,pred_NB)
recall_NB = recall_score(labels_test,pred_NB)
print('Precisão_NB SEM NOVA FEATURES = %0.3f' % precision_NB)
print('Recall_NB SEM NOVA FEATURES = %0.3f' % recall_NB)


################################

# SVM

param_grid_svm = {
             'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.001, 0.0005, 0.0001, 0.005, 0.01, 0.1],
              }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf_svm = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid_svm)
clf_svm = clf_svm.fit(features_train, labels_train)
print ("Melhores etimadores para SVM SEM NOVA FEATURES:")
print (clf_svm.best_estimator_)

pred_svm = clf_svm.predict(features_test)
acc_svm = accuracy_score(pred_svm, labels_test)
print('Acurácia_SVM SEM NOVA FEATURES = %0.3f' % acc_svm)


precision_svm = precision_score(labels_test,pred_svm)
recall_svm = recall_score(labels_test,pred_svm)
print('Precisão_SVM SEM NOVA FEATURES = %0.3f' % precision_svm)
print('Recall_SVM SEM NOVA FEATURES = %0.3f' % recall_svm)

#################################



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

# Exibir os gráficos
plt.show()

